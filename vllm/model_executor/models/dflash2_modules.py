# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The two things a DFlash2 draft adds to a DFlash one.

A grouped dynamic convolution along the block, wrapping each sublayer, and a
direct-A/B candidate selector that scores the transitions between adjacent
proposal slots instead of committing to each slot's argmax independently. Both
are inert on a plain DFlash checkpoint, which declares neither.
"""

import torch
from torch import nn


class DFlashGroupedConv(nn.Module):
    """Grouped dynamic depthwise K-tap convolution across one DFlash block.

    Each sublayer is wrapped: `prepare` convolves its input and returns the kernel
    for `finish` to convolve its output, both from one projection of the input.
    """

    def __init__(
        self, hidden_size: int, block_size: int, taps: int, group_size: int
    ) -> None:
        super().__init__()
        if hidden_size % group_size:
            raise ValueError(
                f"conv_group_size={group_size} must divide hidden_size={hidden_size}"
            )
        if block_size & (block_size - 1):
            raise ValueError(f"block_size={block_size} must be a power of two")
        self.hidden_size = int(hidden_size)
        self.block_size = int(block_size)
        self.taps = int(taps)
        self.group_size = int(group_size)
        self.num_groups = self.hidden_size // self.group_size
        # [input/output, tap, channel], the layout training exports.
        self.base_kernel = nn.Parameter(torch.empty(2, self.taps, self.hidden_size))
        self.kernel_projection = nn.Linear(
            self.hidden_size, 2 * self.taps * self.num_groups, bias=False
        )

    def _convolve(
        self, hidden_states: torch.Tensor, delta: torch.Tensor, side: int
    ) -> torch.Tensor:
        # The token axis stays flat. Splitting it into (blocks, block_size) makes
        # the index arithmetic symbolic on the one axis @support_torch_compile
        # marks dynamic, which nothing downstream can fold; the block boundary is
        # a mask over positions instead.
        blocks = hidden_states.unflatten(-1, (self.num_groups, self.group_size))
        base = self.base_kernel[side].view(
            1, self.taps, self.num_groups, self.group_size
        )
        coefficients = base + delta.unsqueeze(-1)
        out = coefficients[:, 0] * blocks
        position = (
            torch.arange(hidden_states.shape[0], device=hidden_states.device)
            & (self.block_size - 1)
        )
        for tap in range(1, self.taps):
            shifted = torch.nn.functional.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
            keep = (position >= tap).view(-1, 1, 1).to(shifted.dtype)
            out = out + coefficients[:, tap] * shifted * keep
        return out.flatten(-2)

    def prepare(self, hidden_states: torch.Tensor):
        coefficients = self.kernel_projection(hidden_states).reshape(
            *hidden_states.shape[:-1], 2, self.taps, self.num_groups
        )
        return (
            self._convolve(hidden_states, coefficients[..., 0, :, :], side=0),
            coefficients[..., 1, :, :],
        )

    def finish(
        self, hidden_states: torch.Tensor, coefficients: torch.Tensor
    ) -> torch.Tensor:
        return self._convolve(hidden_states, coefficients, side=1)


class CandidateSelector(nn.Module):
    """Scores the K x K transitions between adjacent proposal slots, then walks them.

    The [vocab, r] tables are replicated on every TP rank rather than sharded like
    the LM head: candidate ids are gathered globally, so any rank can need any row.
    """

    def __init__(
        self, *, hidden_size: int, vocab_size: int, state_rank: int, top_k: int
    ) -> None:
        super().__init__()
        self.state_rank = int(state_rank)
        self.top_k = int(top_k)
        self.predecessor_codebook = nn.Embedding(int(vocab_size), self.state_rank)
        self.successor_codebook = nn.Embedding(int(vocab_size), self.state_rank)
        self.predecessor_codebook.weight.requires_grad_(False)
        self.successor_codebook.weight.requires_grad_(False)
        self.hidden_projection = nn.Linear(hidden_size, state_rank, bias=False)

    # Compiled here rather than by the model's @support_torch_compile: the selector
    # runs in the speculator, past what that decorator covers. Inside the captured
    # draft graph this is still worth 61-67% of the two steps together, because a
    # replay pays per node.
    @torch.compile(dynamic=True)
    def score_edges(
        self,
        *,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """score[b,l,p,c] = unary[b,l,c] + <A[pred[b,l,p]] * project(h[b,l]), B[c]>

        pred is cand[b,l-1], and the verified anchor for slot 0.
        """
        predecessor = self.predecessor_codebook.weight
        keys = self.successor_codebook.weight[candidate_ids]
        hidden = self.hidden_projection(hidden_states)
        candidates = predecessor[candidate_ids]
        anchor = predecessor[anchor_token_ids]
        predecessors = torch.cat(
            [anchor[:, None, None].expand(-1, 1, self.top_k, -1), candidates[:, :-1]],
            dim=1,
        )
        return unary_logits[:, :, None] + torch.einsum(
            "blpr,blcr->blpc", predecessors * hidden[:, :, None], keys
        )

    @staticmethod
    @torch.compile(dynamic=True)
    def walk(candidate_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Greedy walk: slot 0 from the anchor edge, then each slot's argmax given
        the slot chosen before it.

        Sequential over the L slots, each an argmax over K=16, which compiles to a
        cost that does not move with batch -- the per-edge maps and a log-depth scan
        would buy nothing here.
        """
        length = scores.shape[1]
        slot = scores[:, 0, 0].argmax(dim=-1)
        slots = [slot]
        for position in range(1, length):
            rows = scores[:, position].gather(
                1, slot[:, None, None].expand(-1, 1, scores.shape[-1])
            )[:, 0]
            slot = rows.argmax(dim=-1)
            slots.append(slot)
        path = torch.stack(slots, dim=1)
        return candidate_ids.gather(-1, path[..., None])[..., 0]

    @staticmethod
    def rows_along(scores: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        """The K-wide score row each slot was chosen from, [B, L, K]: what a
        non-greedy verify needs, conditioned on the token drawn before it."""
        first = scores[:, :1, 0]
        rest = scores[:, 1:].gather(
            2, slots[:, :-1, None, None].expand(-1, -1, 1, scores.shape[-1])
        )[:, :, 0]
        return torch.cat((first, rest), dim=1)

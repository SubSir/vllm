# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The two things a DFlash2 draft adds to a DFlash one.

A grouped dynamic convolution along the block, wrapping each sublayer, and a
direct-A/B candidate selector that scores the transitions between adjacent
proposal slots instead of committing to each slot's argmax independently.

Both are inert on a plain DFlash checkpoint: the convolution is built only when
``dflash_config.conv_type`` says so, and the selector only when
``dflash_config.dflashv2_selector`` is present.
"""

import torch
from torch import nn


class DFlashGroupedConv(nn.Module):
    """Grouped dynamic depthwise K-tap convolution across one DFlash block.

        row_i <- sum_t (base[t] + delta_i[t]) * row_{i-t}

    ``base`` is a static per-channel kernel; ``delta`` is predicted for the row by
    one projection and shared by every channel of a group, so a hidden of H with
    group size g carries H/g coefficients per tap instead of H. One projection
    produces the kernel for the sublayer's input and the one for its output, which
    is why ``prepare`` hands the second half to ``finish``.

    Taps read backwards within the block and are zero across its boundary, so a row
    never sees a position the draft has not proposed yet.

    Written as plain tensor ops rather than a fused kernel: the enclosing model
    carries ``@support_torch_compile``, so inductor fuses this chain the way a
    hand-written kernel would. Unfused, the same expression costs roughly half a
    draft forward -- almost all of it materialising ``base + delta`` and a padded
    copy that a fused form keeps in registers.
    """

    def __init__(
        self, hidden_size: int, block_size: int, taps: int, group_size: int
    ) -> None:
        super().__init__()
        if hidden_size % group_size:
            raise ValueError(
                f"conv_group_size={group_size} must divide hidden_size={hidden_size}"
            )
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
        # No branch on the row count: the enclosing model is compiled with a
        # symbolic batch dimension, and a Python test against it is a guard
        # Dynamo cannot resolve. Every draft forward, dummy runs included, carries
        # whole blocks -- num_query_per_req tokens per request -- so the reshape
        # below is always exact.
        blocks = hidden_states.view(
            -1, self.block_size, self.num_groups, self.group_size
        )
        delta = delta.reshape(-1, self.block_size, self.taps, self.num_groups, 1)
        base = self.base_kernel[side].view(
            1, 1, self.taps, self.num_groups, self.group_size
        )
        coefficients = base + delta
        out = coefficients[:, :, 0] * blocks
        for tap in range(1, self.taps):
            shifted = torch.nn.functional.pad(
                blocks[:, :-tap], (0, 0, 0, 0, tap, 0)
            )
            out = out + coefficients[:, :, tap] * shifted
        return out.view_as(hidden_states)

    def prepare(self, hidden_states: torch.Tensor):
        """Convolve a sublayer's input; return it with the kernel for its output."""
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
    """Direct-edge candidate selector.

    An edge between a predecessor token p at one proposal slot and a candidate c at
    the next is scored directly in the two token directions:

        edge(p -> c) = <A[p] * project(h), B[c]>

    A and B are separate [vocab, r] tables, so predecessor and successor are untied.
    Training folds the 1/sqrt(r) scale into B and ships both materialised, so this
    side only gathers rows; they are replicated rather than vocab-sharded because
    candidate ids are global.
    """

    def __init__(
        self, *, hidden_size: int, vocab_size: int, state_rank: int, top_k: int
    ) -> None:
        super().__init__()
        self.state_rank = int(state_rank)
        self.top_k = int(top_k)
        self.predecessor_token_table = nn.Parameter(
            torch.empty(int(vocab_size), self.state_rank), requires_grad=False
        )
        self.successor_token_table = nn.Parameter(
            torch.empty(int(vocab_size), self.state_rank), requires_grad=False
        )
        self.hidden_projection = nn.Linear(hidden_size, state_rank, bias=False)

    def score_edges(
        self,
        *,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """[B, L, K] candidates -> [B, L, previous_K, current_K] edge scores:

            score[b,l,p,c] = unary[b,l,c] + <A[pred[b,l,p]] * project(h[b,l]), B[c]>

        pred is cand[b,l-1]; slot 0's predecessor is the verified anchor, broadcast
        over p so it needs no code path of its own.
        """
        keys = self.successor_token_table[candidate_ids]
        hidden = self.hidden_projection(hidden_states)
        candidates = self.predecessor_token_table[candidate_ids]
        anchor = self.predecessor_token_table[anchor_token_ids]
        predecessors = torch.cat(
            [anchor[:, None, None].expand(-1, 1, self.top_k, -1), candidates[:, :-1]],
            dim=1,
        )
        return unary_logits[:, :, None] + torch.einsum(
            "blpr,blcr->blpc", predecessors * hidden[:, :, None], keys
        )

    @staticmethod
    def walk(candidate_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Greedy walk: pick slot 0 from the anchor edge, then follow each slot's
        argmax given the slot chosen before it.

        Sequential over the L proposal slots, but each step is an argmax over K=16,
        so the whole walk is a handful of tiny kernels. The alternative -- composing
        the per-edge maps with a log-depth scan -- is only worth its buffers when L
        is much larger than this.
        """
        batch, length = scores.shape[0], scores.shape[1]
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
        """The K-wide score row each slot was actually chosen from, [B, L, K].

        This is what a non-greedy verify needs: the distribution the draft sampled
        from at every position, which is conditioned on the token it drew before.
        """
        first = scores[:, :1, 0]
        rest = scores[:, 1:].gather(
            2, slots[:, :-1, None, None].expand(-1, -1, 1, scores.shape[-1])
        )[:, :, 0]
        return torch.cat((first, rest), dim=1)

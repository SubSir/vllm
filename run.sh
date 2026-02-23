export VLLM_DFLASH_K_ONLINE=1
export VLLM_DFLASH_K_ONLINE_OFFSET=2
export VLLM_DFLASH_K_ONLINE_WARMUP=0

vllm serve Qwen/Qwen3-8B \
  --speculative-config '{"method":"dflash","model":"z-lab/Qwen3-8B-DFlash-b16","num_speculative_tokens":16}' \
  --max-model-len 2048 \
  --max-num-seqs 32 \
  --tensor-parallel-size 2


vllm bench serve \
  --model Qwen/Qwen3-8B \
  --dataset-name hf --dataset-path likaixin/InstructCoder \
  --num-prompts 2048

#!/usr/bin/env bash
# Example commands for ReCalKV. Edit MODEL to point at your local HF checkpoint.
# Datasets are loaded from ./data (run `python prepare_data.py` first, or set
# RECALKV_DATA_ROOT to a custom location).
set -e

MODEL=/path/to/Llama-2-7b-hf          # e.g. NousResearch/Llama-2-7b-hf downloaded locally
RATIO=0.5                             # target KV-cache compression ratio (0.5 = 50%)
GS=4                                  # head group size for grouped SVD

# ---------------------------------------------------------------------------
# 0. Prepare datasets (once)
# ---------------------------------------------------------------------------
# python prepare_data.py --data_root ./data --datasets wikitext,ptb,c4

# ---------------------------------------------------------------------------
# 1. Compress: ReCalKV = HSR (Key) + OVC (Value)
#    decompose_method=ours + search_method=fisher_uniform
#    The compressed HF model is dumped under ./output_model/
# ---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 python compress.py \
    --model_id="${MODEL}" \
    --calib_dataset wikitext2 \
    --decompose_method ours \
    --param_ratio_target "${RATIO}" \
    --search_method fisher_uniform \
    --head_group_size "${GS}" \
    --dump_huggingface_model \
    --use_cache \
    --calib_nsamples 256 \
    --updating_nsamples 256 \
    --updating_dataset wikitext2

COMPRESSED=output_model/$(basename "${MODEL}")_ratio-${RATIO}_gs-${GS}-fisher_uniform-ours_updating_nsamples-256_updating_dataset-wikitext2

# ---------------------------------------------------------------------------
# 2. Perplexity evaluation (WikiText-2 / PTB / C4)
# ---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 python run_ppl_eval.py \
    --model_name_or_path "${COMPRESSED}" \
    --datasets wikitext2,ptb,c4 \
    --seqlen 2048

# With low-rank-aware KV quantization (requires fast-hadamard-transform):
# CUDA_VISIBLE_DEVICES=0 python run_ppl_eval.py \
#     --model_name_or_path "${COMPRESSED}" \
#     --datasets wikitext2,c4 --seqlen 2048 --lt_bits 3 --lt_hadamard

# ---------------------------------------------------------------------------
# 3. Zero-shot accuracy (lm-evaluation-harness)
# ---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 python run_lm_eval.py \
    --model_name_or_path "${COMPRESSED}" \
    --tasks "openbookqa,hellaswag,piqa,arc_easy,arc_challenge,winogrande"

# ---------------------------------------------------------------------------
# 4. LongBench
# ---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 python run_long_bench.py \
    --model_name_or_path "${COMPRESSED}" \
    --datasets "triviaqa,qasper,trec,samsum,lcc,repobench-p,qmsum,multi_news"

# ---------------------------------------------------------------------------
# 5. Latency benchmarks
# ---------------------------------------------------------------------------
# CUDA_VISIBLE_DEVICES=0 python run_latency_attention.py \
#     --rank_k 2048 --rank_v 2048 --group_size 4 --prompt_len 65536 --palu
# CUDA_VISIBLE_DEVICES=0 python run_latency_kernel.py --total_rank 2048 --group_size 4

import argparse
import os

import torch
from loguru import logger
from tqdm import tqdm

from utils import set_seed, dump_to_huggingface_repos, load_model_and_tokenizer
from palu.rank_search import rank_search
from palu.decomposition import compress_model
from run_ppl_eval import eval_ppl


def compress(args):
    set_seed(args.seed)

    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.model_id)

    # Step 1: rank selection -> per-layer compression rate.
    # search_results maps each k/v-proj name to a per-group rank list.
    search_results, rank_sum, total_rank = rank_search(model, tokenizer, args)

    # Step 2: compress the model with the selected ranks.
    compress_model(model, tokenizer, args, args.device, search_results)

    # Quick sanity check on WikiText-2 perplexity.
    results = eval_ppl(model, tokenizer, args.model_id, "wikitext2", 2048, args.device)
    for dataset, ppl in results.items():
        logger.info(f"PPL: {ppl}")

    if args.dump_huggingface_model:
        save_folder = os.path.join(
            "output_model",
            f"{args.model_id.split('/')[-1]}_ratio-{args.param_ratio_target}"
            f"_gs-{args.head_group_size}-{args.search_method}-{args.decompose_method}"
            f"_updating_nsamples-{args.updating_nsamples}_updating_dataset-{args.updating_dataset}",
        )
        dump_to_huggingface_repos(model, tokenizer, save_folder, args)
        logger.info(f"Huggingface model is saved to {save_folder}", fg="green")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-hf",
                        help="Pretrained model ID or local path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dump_huggingface_model", action="store_true",
                        help="Whether to dump the compressed model in HuggingFace format.")
    parser.add_argument("--use_cache", action="store_true",
                        help="Whether to reuse cached Fisher / whitening results.")

    # Calibration data
    parser.add_argument("--calib_dataset", type=str, default="wikitext2",
                        choices=["wikitext2", "c4", "ptb"], help="Calibration dataset for rank search / whitening")
    parser.add_argument("--calib_nsamples", type=int, default=2048, help="Number of calibration samples")
    parser.add_argument("--calib_seqlen", type=int, default=1024, help="Calibration sequence length")
    parser.add_argument("--updating_dataset", type=str, default="wikitext2",
                        choices=["wikitext2", "c4", "ptb"], help="Dataset used for OVC calibration")
    parser.add_argument("--updating_nsamples", type=int, default=256, help="Number of OVC calibration samples")
    parser.add_argument("--model_seq_len", type=int, default=2048, help="Sequence length used during decomposition")

    # Compression hyper-parameters
    parser.add_argument("--param_ratio_target", type=float, default=-1,
                        help="Target KV-cache compression ratio (e.g. 0.5 means 50%%)")
    parser.add_argument("--head_group_size", type=int, default=4,
                        help="Number of heads per group for grouped SVD")
    parser.add_argument("--num_iter", type=int, default=1,
                        help="Number of alternating updates in Offline Value Calibration (OVC)")
    parser.add_argument("--search_method", type=str, default="fisher_uniform",
                        choices=["fisher", "fisher_uniform", "uniform"],
                        help="Rank-allocation method (paper uses fisher_uniform)")
    parser.add_argument("--decompose_method", type=str, default="ours",
                        choices=["ours", "ours_reorder", "ours_calib", "ours_baseline", "whiten"],
                        help="Decomposition method. ours=ReCalKV (HSR+OVC); ours_reorder=HSR only; "
                             "ours_calib=OVC only; ours_baseline=neither; whiten=Palu baseline.")

    parser.add_argument("--verbose", action="store_true", help="Enable verbose (DEBUG) logging.")
    args = parser.parse_args()

    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level="INFO" if not args.verbose else "DEBUG")

    compress(args)

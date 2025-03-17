import os
import time
import json
import sys
import logging
import argparse
from pathlib import Path
from typing import Set
from huggingface_hub import HfApi, list_repo_files
from run_evals import _main as eval_main

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("poller")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Poll a Hugging Face repo for new .pt checkpoints (with 'rank' in filename); call run_evals."
    )
    parser.add_argument("--repo_id", type=str, default="PLACEHOLDER",
                        help="Hugging Face repo ID to monitor for new checkpoints")
    parser.add_argument("--token", type=str, default=None,
                        help="Optional HF API token for private repos")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Local directory to store or download checkpoints; "
                             "this is passed to run_evals._main(..., checkpoints=...)")
    parser.add_argument("--poll_interval", type=int, default=60,
                        help="How many seconds to wait between polls")

    parser.add_argument("--wandb_project", type=str, default=None,
                        help="Optional W&B project to pass to _main")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Optional W&B entity to pass to _main")
    parser.add_argument("--tasks", nargs="+", default=["mnli"],
                        help="Which tasks to evaluate. Will pass as a list of strings to _main.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 314, 1234],
                        help="Random seeds to pass to _main")
    parser.add_argument("--gpu_ids", nargs="+", type=int, default=[3,4,5],
                        help="Optional list of GPU IDs to use for evaluation")
    parser.add_argument("--skip_generation", action="store_true",
                        help="If set, pass skip_generation=True to _main")
    return parser.parse_args()


def load_processed(file_path: str = "processed_checkpoints.json") -> Set[str]:
    """
    Load a set of checkpoint filenames we've already processed, so we don't re‐process them.
    """
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                return set(json.load(f))
        except Exception as e:
            logger.warning(f"Could not parse {file_path}: {e}")
    return set()


def save_processed(processed: Set[str], file_path: str = "processed_checkpoints.json"):
    """
    Save a set of checkpoint filenames, so next time we skip them.
    """
    try:
        with open(file_path, "w") as f:
            json.dump(list(processed), f)
    except Exception as e:
        logger.warning(f"Could not write to {file_path}: {e}")


def find_new_checkpoints(files_in_repo: list[str], processed: Set[str]) -> Set[str]:
    """
    Return any .pt filenames containing 'rank' that are not yet in 'processed'.
    E.g. 'my_run/epoch3-rank0.pt'
    """
    new_ckpts = set()
    for f in files_in_repo:
        if f.endswith(".pt") and "rank" in f and f not in processed:
            new_ckpts.add(f)
    return new_ckpts


def poll_loop(args):
    """
    Main polling loop: 
      - check the HF repo for new .pt files
      - pass them to run_evals.programmatic_main
      - record them in JSON
      - sleep
    """
    hf_api = HfApi(token=args.token)
    processed = load_processed()

    logger.info(f"Starting poller for {args.repo_id}")
    logger.info(f"Polling every {args.poll_interval} seconds.\n")

    while True:
        try:
            logger.info(f"Checking for new checkpoints in {args.repo_id}...")
            repo_files = list_repo_files(args.repo_id, token=args.token)
            new_ckpts = find_new_checkpoints(repo_files, processed)

            if not new_ckpts:
                logger.info("No new checkpoints found.")
            else:
                for ckpt in new_ckpts:
                    logger.info(f"Found new checkpoint: {ckpt}")
                    logger.info("Calling run_evals.programmatic_main(...) on that checkpoint...")

                    try:
                        eval_main(
                            checkpoints=args.checkpoint_dir,
                            hub_repo=args.repo_id,
                            hub_files=[ckpt],
                            hub_token=args.token,
                            wandb_project=args.wandb_project,
                            wandb_entity=args.wandb_entity,
                            tasks=args.tasks,
                            seeds=args.seeds,
                            skip_generation=args.skip_generation,
                            gpu_ids=args.gpu_ids,
                            verbose=True,
                            parallel=True,
                        )
                        # Mark it processed
                        processed.add(ckpt)
                        save_processed(processed)
                    except Exception as e:
                        logger.error(f"Error running eval on {ckpt}: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Error in poll loop: {e}", exc_info=True)

        logger.info(f"Sleeping {args.poll_interval} seconds...\n")
        time.sleep(args.poll_interval)


def main():
    args = parse_args()
    poll_loop(args)


if __name__ == "__main__":
    main()

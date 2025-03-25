import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Annotated, List, Optional, Set

import typer
import yaml
from huggingface_hub import HfApi, list_repo_files
from typer import Option

from run_evals import main as eval_main
from run_evals import TaskName

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("poller")

app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]}, pretty_exceptions_show_locals=False)


# from maxb2: https://github.com/tiangolo/typer/issues/86#issuecomment-996374166
def conf_callback(ctx: typer.Context, param: typer.CallbackParam, config: Optional[str] = None):
    if config is not None:
        typer.echo(f"Loading config file: {config}\n")
        try:
            with open(config, "r") as f:  # Load config file
                conf = yaml.safe_load(f)
            ctx.default_map = ctx.default_map or {}  # Initialize the default map
            ctx.default_map.update(conf)  # Merge the config dict into default_map
        except Exception as ex:
            raise typer.BadParameter(str(ex))
    return config


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


def poll_loop(
    repo_id: str,
    token: Optional[str],
    checkpoint_dir: str,
    poll_interval: int,
    wandb_project: Optional[str],
    wandb_entity: Optional[str],
    tasks: List[str],
    seeds: List[int],
    gpu_ids: List[int],
    skip_generation: bool,
):
    """
    Main polling loop:
      - check the HF repo for new .pt files
      - pass them to run_evals.programmatic_main
      - record them in JSON
      - sleep
    """
    hf_api = HfApi(token=token)
    processed = load_processed()

    logger.info(f"Starting poller for {repo_id}")
    logger.info(f"Polling every {poll_interval} seconds.\n")

    while True:
        try:
            logger.info(f"Checking for new checkpoints in {repo_id}...")
            repo_files = list_repo_files(repo_id, token=token)
            new_ckpts = find_new_checkpoints(repo_files, processed)

            if not new_ckpts:
                logger.info("No new checkpoints found.")
            else:
                for ckpt in new_ckpts:
                    logger.info(f"Found new checkpoint: {ckpt}")
                    logger.info("Calling run_evals.programmatic_main(...) on that checkpoint...")

                    try:
                        eval_main(
                            checkpoints=checkpoint_dir,
                            hub_repo=repo_id,
                            hub_files=[ckpt],
                            hub_token=token,
                            wandb_project=wandb_project,
                            wandb_entity=wandb_entity,
                            tasks=tasks,
                            seeds=seeds,
                            skip_generation=skip_generation,
                            gpu_ids=gpu_ids,
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

        logger.info(f"Sleeping {poll_interval} seconds...\n")
        time.sleep(poll_interval)


@app.command()
def main(
    repo_id: Annotated[str, Option(help="Hugging Face repo ID to monitor for new checkpoints", show_default=False)],
    token: Annotated[Optional[str], Option(help="Optional HF API token for private repos")] = None,
    checkpoint_dir: Annotated[Path, Option(help="Local directory to store or download checkpoints")] = "./checkpoints",
    poll_interval: Annotated[int, Option(help="How many seconds to wait between polls")] = 60,
    wandb_project: Annotated[Optional[str], Option(help="Optional W&B project to pass to eval script")] = None,
    wandb_entity: Annotated[Optional[str], Option(help="Optional W&B entity to pass to eval script")] = None,
    tasks: Annotated[List[TaskName], Option(help="Which tasks to evaluate")] = [TaskName.mnli], # type: ignore
    seeds: Annotated[List[int], Option(help="Random seeds to pass to _main")] = [42, 314, 1234],
    gpu_ids: Annotated[Optional[List[int]], Option(help="Optional list of GPU IDs to use for evaluation")] = None,
    skip_generation: Annotated[bool, Option(help="If set, pass skip_generation=True to eval script")] = False,
    config: Annotated[Optional[Path], Option(callback=conf_callback, is_eager=True, help="Relative path to YAML config file for setting options. Passing CLI options will supersede config options.", case_sensitive=False, rich_help_panel="Options")] = None,
):  # fmt: skip
    """
    Poll a Hugging Face repo for new .pt checkpoints (with 'rank' in filename); call run_evals.
    """
    poll_loop(
        repo_id=repo_id,
        token=token,
        checkpoint_dir=checkpoint_dir,
        poll_interval=poll_interval,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        tasks=tasks,
        seeds=seeds,
        gpu_ids=gpu_ids,
        skip_generation=skip_generation,
    )


if __name__ == "__main__":
    app()

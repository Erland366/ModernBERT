# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

import os
import random
import re
import signal
import subprocess
import tempfile
import time
import warnings
from collections import deque
from enum import Enum
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Annotated, List, Optional, Union

import datasets
import psutil
import typer
import yaml
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from typer import Exit, Option

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    from eval import GLUE_TASKS, SUPERGLUE_TASKS, TASK_NAME_TO_CLASS


# Create TaskName enum dynamically from TASK_NAME_TO_CLASS keys
TaskName = Enum("TaskName", {name: name for name in TASK_NAME_TO_CLASS.keys()}, type=str)


app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]}, pretty_exceptions_show_locals=False)


class ModelSize(str, Enum):
    BASE = "base"
    LARGE = "large"
    HUGE = "huge"

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


# Global dictionary to keep track of GPUs with running jobs
# Changed to store more information per GPU
gpus_in_use = {}
# Queue to keep track of GPUs that might be free
potentially_free_gpus = deque()
# Global list to keep track of all running processes
all_processes = []

# Global list to specify which GPUs to use
allowed_gpus = None  # Will be set to list of GPU IDs or None

console = Console()

def kill_process_tree(pid: int):
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            child.terminate()
        gone, still_alive = psutil.wait_procs(children, timeout=5)
        for p in still_alive:
            p.kill()
        parent.terminate()
        parent.wait(5)
    except psutil.NoSuchProcess:
        pass


def signal_handler(signum, frame):
    print("\nReceived termination signal. Cleaning up subprocesses...")
    for process in all_processes:
        if process.poll() is None:  # If the process is still running
            kill_process_tree(process.pid)

    print("Cleanup completed. Exiting.")
    os._exit(0)  # Force exit without running cleanup handlers


def get_gpu_memory_usage(gpu_id):
    """Get memory usage for a specific GPU."""
    try:
        output = (
            subprocess.check_output(
                f"nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader -i {gpu_id}", shell=True
            )
            .decode("utf-8")
            .strip()
        )
        return int(output)
    except subprocess.CalledProcessError:
        print(f"Failed to get memory usage for GPU {gpu_id}")
        return None


def get_free_gpu():
    """Check for free GPUs, prioritizing potentially free GPUs."""
    global allowed_gpus
    while potentially_free_gpus:
        gpu_id = potentially_free_gpus.popleft()
        if (allowed_gpus is None or gpu_id in allowed_gpus) and gpu_id not in gpus_in_use:
            memory_used = get_gpu_memory_usage(gpu_id)
            if memory_used is not None and memory_used < 100:
                return gpu_id

    # If no potentially free GPUs, check allowed GPUs
    try:
        gpu_output = subprocess.check_output(
            "nvidia-smi --query-gpu=index,memory.used --format=csv,nounits,noheader", shell=True
        ).decode("utf-8")
        for line in gpu_output.strip().split("\n"):
            gpu_id, memory_used = map(int, line.split(","))
            if (allowed_gpus is None or gpu_id in allowed_gpus) and memory_used < 100 and gpu_id not in gpus_in_use:
                return gpu_id
        return None
    except subprocess.CalledProcessError:
        print("Failed to execute nvidia-smi")
        return None


def run_subprocess(cmd: List[str], verbose: bool = False, show_errors: bool = False):
    stdout = None if verbose else subprocess.DEVNULL
    stderr = None if verbose or show_errors else subprocess.DEVNULL
    process = subprocess.Popen(cmd, stdout=stdout, stderr=stderr)
    all_processes.append(process)  # Add the process to the global list
    process.wait()


def handle_process_completion(process, stderr_file, config_path: Path, verbose: bool, gpu_id: Optional[int] = None):
    """Handles the completion of a process, checks for errors, cleans up stderr_file, and logs messages."""
    returncode = process.returncode

    # Read and clean up stderr output
    if stderr_file is not None:
        stderr_file.seek(0)
        error_output = stderr_file.read()
        stderr_file.close()
        os.unlink(stderr_file.name)  # Delete the temp file
    else:
        error_output = "Error output was displayed above."

    # Construct job identifier
    if gpu_id is not None:
        job_identifier = f"Job on GPU {gpu_id} for {config_path.name}"
    else:
        job_identifier = f"Job for {config_path.name}"

    if returncode != 0:
        # The process exited with an error
        if verbose:
            print(f"{job_identifier} failed with return code {returncode}")
            print("Error Output:")
            print(error_output)
        else:
            console.print(f"[red]{job_identifier} failed with return code {returncode}[/red]")
            console.print(f"[red]Error Output:[/red]\n{error_output}")
    else:
        # The process completed successfully
        if verbose:
            print(f"{job_identifier} has finished successfully.")
        else:
            console.log(f"{job_identifier} has finished successfully.")


def run_job(
    config_path: Path,
    verbose: bool = False,
    delete_eval_yamls: bool = True,
    gpu_id: Optional[int] = None,
    gpu_ids: Optional[List[int]] = None,
):
    """Run a job with optional GPU management."""
    if gpu_id is not None:
        # GPU management is required
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    elif gpu_ids is not None:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    else:
        env = None  # Use default environment

    if verbose:
        stdout = None  # Output will be shown directly
        stderr = None
        stderr_file = None
    else:
        stdout = subprocess.DEVNULL
        stderr_file = tempfile.NamedTemporaryFile(mode="w+", delete=False)
        stderr = stderr_file

    process = subprocess.Popen(["python", "eval.py", str(config_path)], env=env, stdout=stdout, stderr=stderr)
    all_processes.append(process)  # Add the process to the global list

    if gpu_id is not None:
        # Store process info for GPU management
        gpus_in_use[gpu_id] = {"process": process, "stderr_file": stderr_file, "config": config_path}

    else:
        process.wait()
        handle_process_completion(process, stderr_file, config_path, verbose, gpu_id=None)
        if delete_eval_yamls:
            config_path.unlink()

    return process


def check_finished_jobs(verbose: bool = False):
    """Check for finished jobs and free up their GPUs."""
    finished_gpus = []
    for gpu_id, info in gpus_in_use.items():
        process = info["process"]
        stderr_file = info["stderr_file"]
        config = info["config"]

        if process.poll() is not None:  # Job has finished
            # Handle process completion
            handle_process_completion(process, stderr_file, config, verbose, gpu_id=gpu_id)
            finished_gpus.append(gpu_id)

    for gpu_id in finished_gpus:
        del gpus_in_use[gpu_id]
        potentially_free_gpus.append(gpu_id)


def manage_jobs(configs: List[Path], verbose: bool = False, delete_eval_yamls: bool = True):
    """Manage the launching of jobs for each configuration file in the directory."""

    if verbose:
        for config in configs:
            while True:
                check_finished_jobs(verbose)
                gpu_id = get_free_gpu()
                if gpu_id is not None:
                    time.sleep(random.randint(0, 5))
                    print(f"\nLaunching job for {config} on GPU {gpu_id}\n")
                    run_job(config, gpu_id=gpu_id, verbose=verbose, delete_eval_yamls=delete_eval_yamls)
                    break
                else:
                    time.sleep(10)

        # Wait for all remaining jobs to finish
        while gpus_in_use:
            check_finished_jobs(verbose)
            time.sleep(10)
    else:

        def update_progress_for_finished_jobs():
            """Update progress bars for any finished GPU jobs."""
            for gpu_id, info in list(gpus_in_use.items()):
                process = info["process"]
                if process.poll() is not None:  # Job finished
                    if gpu_id in gpu_tasks:
                        gpu_progress.update(gpu_tasks[gpu_id], completed=1, visible=False)
                        completed_configs.add(info["config"])
                        overall_progress.update(overall_task, completed=len(completed_configs))

        overall_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.completed}/{task.total}"),
            TimeElapsedColumn(),
        )

        gpu_progress = Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), TimeElapsedColumn()
        )

        progress_group = Group(
            Panel(overall_progress, title="Overall Progress", border_style="blue", padding=(1, 1)),
            Panel(gpu_progress, title="GPU Jobs", border_style="green", padding=(1, 1)),
        )

        with Live(progress_group, console=console, refresh_per_second=4):
            overall_task = overall_progress.add_task("[cyan]Overall Progress", total=len(configs))
            gpu_tasks = {}
            completed_configs = set()  # Track completed configs

            for config in configs:
                while True:
                    check_finished_jobs(verbose)
                    update_progress_for_finished_jobs()

                    gpu_id = get_free_gpu()
                    if gpu_id is not None:
                        time.sleep(random.randint(0, 5))
                        if gpu_id not in gpu_tasks:
                            gpu_tasks[gpu_id] = gpu_progress.add_task(f"[green]GPU {gpu_id}", total=1)
                        else:
                            gpu_progress.update(gpu_tasks[gpu_id], completed=1, visible=False)
                            gpu_tasks[gpu_id] = gpu_progress.add_task(f"[green]GPU {gpu_id}", total=1)
                        gpu_progress.update(gpu_tasks[gpu_id], description=f"[green]GPU {gpu_id}: {config.name}")
                        run_job(config, gpu_id=gpu_id, verbose=verbose, delete_eval_yamls=delete_eval_yamls)
                        break
                    else:
                        time.sleep(10)

            # Wait for all remaining jobs to finish
            while gpus_in_use:
                check_finished_jobs(verbose)
                update_progress_for_finished_jobs()
                time.sleep(10)

            overall_progress.update(overall_task, completed=len(configs))

    if delete_eval_yamls:
        for config in configs:
            try:
                config.unlink()
            except FileNotFoundError:
                pass


def create_symlink_for_newest_checkpoint(folder: Path, override_existing: bool = False):
    """Create a symlink to the newest checkpoint file if 'latest-rank0.pt' does not exist."""
    if folder.is_dir():
        pt_files = list(folder.glob("*.pt"))
        if not pt_files:
            print(f"   Warning: No .pt file found in {folder}, skipping symlink creation.")
            return

        if len(pt_files) == 1 and pt_files[0].name == "latest-rank0.pt" and not pt_files[0].is_symlink():
            print(f"   Only found one .pt in {folder.name}, named 'latest-rank0.pt' (real file). Skipping symlink creation.")
            return

        # Sort files based on epoch and batch numbers extracted from filenames
        def extract_numbers(filename: Path):
            if filename.is_symlink():
                return (0, 0)
            if filename.name == "latest-rank0.pt":
                return (0, 0)

            try:
                # Using regex to find patterns of 'ep' followed by digits and 'ba' followed by digits
                match = re.search(r"ep(\d+)-ba(\d+)", filename.stem)
                if match:
                    epoch, batch = map(int, match.groups())
                    return (epoch, batch)
                else:
                    raise ValueError(f"Filename does not match expected pattern: {filename}")
            except Exception as e:
                print(f"   Error extracting numbers from filename {filename}: {e}")
                return (0, 0)

        newest_file = max(pt_files, key=extract_numbers)

        symlink_path = folder / "latest-rank0.pt"
        if symlink_path.is_symlink():
            if symlink_path.resolve() == newest_file.resolve():
                print(f"   Existing symlink in {folder.name} already points to {newest_file.name}")
                return
            else:
                print(
                    f"   Warning: symlink in {folder.name} points to {symlink_path.resolve().name}, "
                    f"but newest is {newest_file.name}"
                )
                if not override_existing:
                    return
                symlink_path.unlink(missing_ok=True)
        elif symlink_path.exists():
            if not override_existing:
                print(f"   {symlink_path.name} is a real file in {folder.name}. Use override to remove it.")
                return
            symlink_path.unlink(missing_ok=True)

        symlink_path.symlink_to(newest_file.name)
        if override_existing:
            print(f"   Overwrote symlink {symlink_path.name} -> {newest_file.name}")
        else:
            print(f"   Created new symlink {symlink_path.name} -> {newest_file.name}")


def generate_eval_configs(
    checkpoints: Path,
    train_config: Optional[Path],
    wandb_run: Optional[str],
    wandb_project: Optional[str],
    wandb_entity: Optional[str],
    track_run: bool,
    track_run_project: Optional[str],
    pooling_type: Optional[str],
    head_class_act: Optional[str],
    head_class_norm: Optional[str],
    head_class_dropout: float,
    tasks: Optional[List[Union[TaskName, str]]],  # type: ignore
    fast_ultrafeedback: bool,
    seeds: List[int],
    parallel: bool,
    use_dir_names: Optional[bool],
    model_size: ModelSize,
    rope_theta: Optional[float],
    gpu_ids: Optional[List[int]] = None,
):
    """Generate evaluation configs for each checkpoint."""

    folders = [
        folder
        for folder in checkpoints.glob("*")
        if folder.is_dir()
        and not folder.name.startswith(".")
        and any(file.suffix == ".pt" for file in folder.glob("*.pt"))
    ]
    if use_dir_names is None and len(folders) > 1:
        use_dir_names = True
        print("Using folder names as run names since multiple `checkpoints` were provided with one `train_config`.")

    for folder in folders:
        cmd = [
            "python",
            "generate_eval_config.py",
            "--checkpoint",
            str(folder),
            "--output-dir",
            str(checkpoints),
        ]

        # Add optional arguments if they're provided
        if use_dir_names:
            cmd.append("--use-dir-name")
        if model_size:
            cmd.extend(["--model-size", model_size.value])
        if rope_theta is not None:
            cmd.extend(["--rope-theta", str(rope_theta)])
        if train_config:
            cmd.extend(["--train-config", str(train_config)])
        if wandb_run:
            cmd.extend(["--wandb-run", wandb_run])
        if wandb_project:
            cmd.extend(["--wandb-project", wandb_project])
        if wandb_entity:
            cmd.extend(["--wandb-entity", wandb_entity])
        if track_run:
            cmd.append("--track-run")
            if track_run_project:
                cmd.extend(["--track-run-project", track_run_project])

        # Classification head options
        if pooling_type:
            cmd.extend(["--pooling-type", pooling_type])
        if head_class_act:
            cmd.extend(["--head-class-act", head_class_act])
        if head_class_norm:
            cmd.extend(["--head-class-norm", head_class_norm])
        if head_class_dropout > 0:
            cmd.extend(["--head-class-dropout", str(head_class_dropout)])

        # Add tasks
        if tasks:
            for task in tasks:
                if hasattr(task, "value"):
                    cmd.extend(["--tasks", task.value])
                else:
                    cmd.extend(["--tasks", str(task)])

        if fast_ultrafeedback:
            cmd.append("--fast-ultrafeedback")

        for seed in seeds:
            cmd.extend(["--seeds", str(seed)])

        if parallel:
            cmd.append("--parallel")

        if gpu_ids:
            if isinstance(gpu_ids, int): gpu_ids = [gpu_ids]
            for g in gpu_ids: cmd.extend(["--gpu-ids", str(g)])

        # Run the config generation process without suppressing output

        run_subprocess(cmd, show_errors=True)
        if not train_config:
            time.sleep(1)


def download_dataset(dataset_name: str, subset: Optional[str] = None):
    try:
        datasets.load_dataset(dataset_name, subset, trust_remote_code=True)
        return f"Successfully downloaded {dataset_name} {subset}"
    except Exception as e:
        return f"Error in processing {dataset_name}: {e}"


def download_datasets(tasks: List[Union[TaskName, str]], msg_queue):  # type: ignore
    try:
        required_datasets = []
        task_to_datasets = {
            "mlmmlu_amateur_semipro": [["answerdotai/MLMMLU", "Amateur"], ["answerdotai/MLMMLU", "Semipro"]],
            "mlmmlu_rookie_reserve": [["answerdotai/MLMMLU", "Rookie"], ["answerdotai/MLMMLU", "Reserve"]],
            "eurlex": [["coastalcph/lex_glue", "eurlex"]],
            "ultrafeedback": [["rbiswasfc/ultrafeedback-binary-classification"]],
        }
        for t in tasks:
            if hasattr(t, "value"):
                task_val = t.value
            else:
                task_val = str(t)

            if task_val in GLUE_TASKS:
                required_datasets.append(["glue", task_val])
            elif task_val in SUPERGLUE_TASKS:
                required_datasets.append(["aps/super_glue", task_val])
            else:
                extras = task_to_datasets.get(task_val, [])
                required_datasets.extend(extras)

        # Suppress output globally in this process
        import sys

        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

        msgs = []
        for dataset_name, subset in required_datasets:
            datasets.load_dataset(dataset_name, subset, trust_remote_code=True)
            msgs.append(f"Successfully downloaded {dataset_name} {subset}")
        msg_queue.put("    " + "\n    ".join(msgs) + "\n")
    except Exception as e:
        msg_queue.put(f"Error in downloading datasets: {e}")


def find_checkpoint_file(file_path: str, repo_files: List[str]) -> Optional[str]:
    import re

    # Filter files in the specified file_path that end with .pt or .yaml
    valid_files = [file for file in repo_files if file.startswith(file_path) and file.endswith((".pt", ".yaml"))]

    if len(valid_files) == 1:
        return valid_files[0]

    # Function to extract epoch and batch numbers from the filename
    def extract_numbers(filename: str):
        match = re.search(r"ep(\d+)-ba(\d+)", filename)
        if match:
            epoch, batch = map(int, match.groups())
            return epoch, batch
        return -1, -1  # Return a default value for files that don't match the pattern

    # Find the newest file based on epoch and batch numbers
    newest_file = max(valid_files, key=extract_numbers, default=None)

    return newest_file


def download_hub_files(
    repo_id: str,
    filenames: Optional[List[str]],
    output_dir: Path,
    repo_type: str = "model",
    token: Optional[str] = None,
) -> List[Path]:
    """Download specific files or the entire repository from a Hugging Face Hub repository."""
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded_files = []

    def move_and_flatten_files(local_dir: Path):
        for file_path in local_dir.rglob("*"):
            if file_path.is_file() and file_path.name.endswith((".pt", ".yaml")):
                # Determine the target directory
                target_dir = output_dir / file_path.parent.name

                # Check if the file is already in the correct location
                if file_path.parent.resolve() in [target_dir.resolve(), output_dir.resolve()]:
                    downloaded_files.append(file_path)
                    continue

                # Create the target directory if it doesn't exist
                target_dir.mkdir(parents=True, exist_ok=True)
                # Move the file to the target directory
                new_path = target_dir / file_path.name
                file_path.rename(new_path)
                downloaded_files.append(new_path)

    # List all files in the repository
    api = HfApi()
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type, token=token)

    try:
        if not filenames:
            # Check if files already exist before downloading entire repository
            existing_files = list(output_dir.glob("**/*.pt")) + list(output_dir.glob("**/*.yaml"))
            if existing_files:
                print(f"Found existing files in '{output_dir}', skipping download.")
                return existing_files

            # Download the entire repository
            local_dir = snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=None,
                cache_dir=None,
                local_dir=output_dir,
                use_auth_token=token,
            )
            move_and_flatten_files(Path(local_dir))
            print(f"Successfully downloaded and flattened the repository '{repo_id}' to '{output_dir}'.")
        else:
            for filename in filenames:
                resolved_filename = find_checkpoint_file(filename, repo_files)
                if not resolved_filename:
                    print(f"Warning: Could not find matching file for '{filename}' in repository.")
                    continue

                # Check if file exists in output_dir or any immediate subdirectory
                filename = Path(resolved_filename).name
                parent_dir = Path(resolved_filename).parent.name
                existing_files = list(output_dir.glob(f"**/{parent_dir}/{filename}"))
                if existing_files:
                    existing_file = existing_files[0]
                    print(f"File '{parent_dir}/{filename}' already exists at '{existing_file}', skipping download.")
                    downloaded_files.append(existing_file)
                    continue

                # Download the file
                _ = hf_hub_download(
                    repo_id=repo_id,
                    filename=resolved_filename,
                    repo_type=repo_type,
                    token=token,
                    local_dir=output_dir,
                    cache_dir=None,
                )
                print(f"Successfully downloaded '{resolved_filename}' from '{repo_id}'.")
            move_and_flatten_files(output_dir)
    except Exception as e:
        print(f"Error downloading from '{repo_id}': {e}")

    return downloaded_files


def _main(
    checkpoints: Union[str, Path],
    train_config: Optional[Union[str, Path]] = None,
    model_size: ModelSize = ModelSize.BASE,
    rope_theta: Optional[float] = None,
    skip_generation: bool = False,
    run_all_yamls: bool = False,
    tasks: Optional[List[Union[str, TaskName]]] = None,
    hub_repo: Optional[str] = None,
    hub_files: Optional[List[str]] = None,
    hub_token: Optional[str] = None,
    wandb_run: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    track_run: bool = False,
    track_run_project: Optional[str] = None,
    pooling_type: Optional[str] = None,
    head_class_act: Optional[str] = None,
    head_class_norm: Optional[str] = None,
    head_class_dropout: float = 0.0,
    fast_ultrafeedback: bool = False,
    seeds: List[int] = [1618, 42, 6033, 3145],
    verbose: bool = False,
    overwrite_existing_symlinks: bool = False,
    parallel: bool = False,
    delete_eval_yamls: bool = False,
    use_dir_names: Optional[bool] = None,
    gpu_ids: Optional[List[int]] = None,
    config: Optional[Union[str, Path]] = None,
):
    if isinstance(checkpoints, str):
        checkpoints = Path(checkpoints)
    if isinstance(train_config, str):
        train_config = Path(train_config)

    global allowed_gpus
    allowed_gpus = gpu_ids

    if hub_repo:
        print(f"\nDownloading from {hub_repo} to {checkpoints} ...")
        downloaded_files = download_hub_files(
            repo_id=hub_repo,
            filenames=hub_files,
            output_dir=checkpoints,
            token=hub_token
        )
        if not downloaded_files:
            print("No files were downloaded successfully. Exiting.")
            raise Exit(code=1)
        print(f"Successfully downloaded {len(downloaded_files)} files to {checkpoints}")

    if not tasks or len(tasks) == 0:
        tasks = [t for t in TaskName]

    print("\nAsynchronously downloading required datasets...")
    msg_queue = Queue()
    download_process = Process(target=download_datasets, args=(tasks, msg_queue))
    download_process.start()

    print("\nCreating symlinks for newest checkpoints...")
    for folder in checkpoints.glob("*"):
        if folder.is_dir() and not folder.name.startswith("."):
            create_symlink_for_newest_checkpoint(folder, overwrite_existing_symlinks)

    if not skip_generation:
        print("\nGenerating evaluation configs...\n")

        if not run_all_yamls:
            config_files_completed = list(checkpoints.glob("*_evaluation.yaml"))
            print("Skipping Completed Jobs (delete yamls to run):")
            for config in config_files_completed:
                print(f"   {config.name}\n")
        else:
            config_files_completed = []

        generate_eval_configs(
            checkpoints=checkpoints,
            train_config=train_config,
            wandb_run=wandb_run,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            track_run=track_run,
            track_run_project=track_run_project,
            pooling_type=pooling_type,
            head_class_act=head_class_act,
            head_class_norm=head_class_norm,
            head_class_dropout=head_class_dropout,
            tasks=tasks,
            fast_ultrafeedback=fast_ultrafeedback,
            seeds=seeds,
            parallel=parallel,
            use_dir_names=use_dir_names,
            model_size=model_size,
            rope_theta=rope_theta,
            gpu_ids=gpu_ids,
        )
        config_files = list(set(checkpoints.glob("*_evaluation.yaml")) - set(config_files_completed))
        config_files = sorted(config_files)
    else:
        config_files = list(checkpoints.glob("*_evaluation.yaml"))

    print("Jobs to be run:")
    for config in config_files:
        print(f"   {config.name}\n")

    # Wait for the dataset download to complete
    print("Waiting for dataset downloads to complete...")
    download_process.join()
    print("\nDataset downloading complete.")
    while not msg_queue.empty():
        print(msg_queue.get())

    if len(config_files) >= 1 and not parallel:
        manage_jobs(config_files, verbose=verbose, delete_eval_yamls=delete_eval_yamls)
    elif len(config_files) > 1 and parallel:
        raise ValueError("Parallel runs only supported for a single config at a time.")
    elif len(config_files) == 1 and parallel:
        if not verbose:
            console.print(f"[bold green]Running {config_files[0].name} in parallel on GPUs: {gpu_ids}")
        run_job(config_files[0], verbose=verbose, delete_eval_yamls=delete_eval_yamls, gpu_ids=gpu_ids)
    else:
        message = "No configuration files found in the specified directory."
        if verbose:
            print(message)
        else:
            console.print(f"[bold red]{message}")

        raise Exit(code=1)

    if verbose:
        print("All jobs completed.")
    else:
        console.print("[bold green]All jobs completed.")


@app.command()
def main(
    checkpoints: Annotated[Path, Option("--checkpoints", help="Directory for model checkpoints.")],
    train_config: Annotated[Optional[Path], Option(help="Path to .yaml config")] = None,
    model_size: Annotated[ModelSize, Option("--model-size")] = ModelSize.BASE,
    rope_theta: Annotated[Optional[float], Option("--rope-theta")] = None,
    skip_generation: Annotated[bool, Option("--skip-generation")] = False,
    run_all_yamls: Annotated[bool, Option("--run-all-yamls")] = False,
    tasks: Annotated[Optional[List[TaskName]], Option(help="Tasks")] = None,
    hub_repo: Annotated[Optional[str], Option("--hub-repo")] = None,
    hub_files: Annotated[Optional[List[str]], Option("--hub-files")] = None,
    hub_token: Annotated[Optional[str], Option("--hub-token")] = None,
    wandb_run: Annotated[Optional[str], Option("--wandb-run")] = None,
    wandb_project: Annotated[Optional[str], Option("--wandb-project")] = None,
    wandb_entity: Annotated[Optional[str], Option("--wandb-entity")] = None,
    track_run: Annotated[bool, Option("--track-run")] = False,
    track_run_project: Annotated[Optional[str], Option("--track-run-project")] = None,
    pooling_type: Annotated[Optional[str], Option("--pooling-type")] = None,
    head_class_act: Annotated[Optional[str], Option("--head-class-act")] = None,
    head_class_norm: Annotated[Optional[str], Option("--head-class-norm")] = None,
    head_class_dropout: Annotated[float, Option("--head-class-dropout")] = 0.0,
    fast_ultrafeedback: Annotated[bool, Option("--fast-ultrafeedback")] = False,
    seeds: Annotated[List[int], Option("--seeds")] = [1618, 42, 6033, 3145],
    verbose: Annotated[bool, Option("-v", "--verbose")] = False,
    overwrite_existing_symlinks: Annotated[bool, Option("--override-existing-symlinks")] = False,
    parallel: Annotated[bool, Option("--parallel")] = False,
    delete_eval_yamls: Annotated[bool, Option("--delete/--keep")] = False,
    use_dir_names: Annotated[Optional[bool], Option("--use-dir-names")] = None,
    gpu_ids: Annotated[Optional[List[int]], Option("--gpu-ids")] = None,
    config: Annotated[Optional[Path], Option(callback=conf_callback, is_eager=True, help="YAML config file")] = None,
):
    _main(
        checkpoints=checkpoints,
        train_config=train_config,
        model_size=model_size,
        rope_theta=rope_theta,
        skip_generation=skip_generation,
        run_all_yamls=run_all_yamls,
        tasks=tasks,
        hub_repo=hub_repo,
        hub_files=hub_files,
        hub_token=hub_token,
        wandb_run=wandb_run,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        track_run=track_run,
        track_run_project=track_run_project,
        pooling_type=pooling_type,
        head_class_act=head_class_act,
        head_class_norm=head_class_norm,
        head_class_dropout=head_class_dropout,
        fast_ultrafeedback=fast_ultrafeedback,
        seeds=seeds,
        verbose=verbose,
        overwrite_existing_symlinks=overwrite_existing_symlinks,
        parallel=parallel,
        delete_eval_yamls=delete_eval_yamls,
        use_dir_names=use_dir_names,
        gpu_ids=gpu_ids,
        config=config,
    )


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    try:
        app()
    finally:
        # Ensure all subprocesses are terminated when the script exits
        for process in all_processes:
            if process.poll() is None:
                process.terminate()
        for process in all_processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

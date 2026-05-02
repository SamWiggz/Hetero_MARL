import os
import sys
from pathlib import Path

import torch
import torch.multiprocessing as mp

import Config


def _prepare_imports():
    root = Path(__file__).resolve().parent
    os.chdir(root)
    root_str = str(root)
    if root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)
    return root


def _validate_layout():
    num_gpu_processes, num_cpu_processes, cores_per_cpu_process = Config.config
    if num_gpu_processes < 0 or num_cpu_processes < 0:
        raise ValueError("Config.config cannot contain negative process counts.")
    if num_gpu_processes + num_cpu_processes < 1:
        raise ValueError("Config.config must request at least one CPU or GPU process.")
    if cores_per_cpu_process < 0:
        raise ValueError("Config.config cannot contain a negative core count.")
    if num_cpu_processes > 0 and cores_per_cpu_process < 1:
        raise ValueError(
            "Config.config must set cores_per_cpu_process to at least 1 "
            "when CPU processes are requested."
        )
    if num_gpu_processes > torch.cuda.device_count():
        raise ValueError(
            "Config.config requests {} GPU process(es), but only {} CUDA device(s) are visible.".format(
                num_gpu_processes, torch.cuda.device_count()
            )
        )


def _selected_env_id_from_cli():
    if len(sys.argv) <= 1:
        return None
    if len(sys.argv) == 2:
        return sys.argv[1]
    raise SystemExit("Usage: python main.py [formation|line|graph_navigation]")


def main():
    _prepare_imports()
    _validate_layout()

    import onpolicy.scripts.train_mpe as train_mpe

    env_id = _selected_env_id_from_cli()
    selected_env_id, _ = Config.get_env_settings(env_id)
    argv = Config.build_argv(selected_env_id)
    world_size = Config.cpu_processes + Config.gpu_process

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = Config.port
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    print("Running InforMARL env_id:", selected_env_id)
    print("Hetero_MARL configuration:", Config.config)

    ctx = mp.get_context("fork")
    processes = [
        ctx.Process(target=train_mpe.main, args=(rank, world_size, argv))
        for rank in range(world_size)
    ]

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    failed = [process.exitcode for process in processes if process.exitcode != 0]
    if failed:
        raise SystemExit("InforMARL worker failed with exit code(s): {}".format(failed))


if __name__ == "__main__":
    main()

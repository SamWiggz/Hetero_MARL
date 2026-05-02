import multiprocessing.shared_memory as shm
import sys

import numpy as np
import torch
import torch.multiprocessing as mp
from gym.spaces import Box

sys.path.append('../')

from utils import make_parallel_env, get_env_config

import Config


def initSharedMemory():
    """
    Initialize shared memory blocks and inter-process parameters.
    """
    env = make_parallel_env(Config.env_id, Config.n_rollout_threads, Config.seed,
                            Config.discrete_action, get_env_config(Config))

    obs_shm = shm.SharedMemory(create=True, size=Config.buffer_length * sum([obsp.shape[0] for obsp in env.observation_space]) * np.dtype(np.float64).itemsize)
    ac_shm = shm.SharedMemory(create=True, size=Config.buffer_length * sum([acsp.shape[0] if isinstance(acsp, Box) else acsp.n for acsp in env.action_space]) * np.dtype(np.float64).itemsize)
    rew_shm = shm.SharedMemory(create=True, size=Config.buffer_length * len(env.agent_types) * np.dtype(np.float64).itemsize)
    next_obs_shm = shm.SharedMemory(create=True, size=Config.buffer_length * sum([obsp.shape[0] for obsp in env.observation_space]) * np.dtype(np.float64).itemsize)
    done_shm = shm.SharedMemory(create=True, size=Config.buffer_length * len(env.agent_types) * np.dtype(np.float64).itemsize)

    env.close()

    curr_i = shm.SharedMemory(create=True, size=np.dtype(np.int32).itemsize)
    filled_i = shm.SharedMemory(create=True, size=np.dtype(np.int32).itemsize)
    t = shm.SharedMemory(create=True, size=np.dtype(np.int32).itemsize)
    ep = shm.SharedMemory(create=True, size=np.dtype(np.int32).itemsize)

    np.ndarray((1,), dtype=np.int32, buffer=curr_i.buf)[0] = 0
    np.ndarray((1,), dtype=np.int32, buffer=filled_i.buf)[0] = 0
    np.ndarray((1,), dtype=np.int32, buffer=t.buf)[0] = 0
    np.ndarray((1,), dtype=np.int32, buffer=ep.buf)[0] = 0

    return {
        'obs_shm': obs_shm,
        'ac_shm': ac_shm,
        'rew_shm': rew_shm,
        'next_obs_shm': next_obs_shm,
        'done_shm': done_shm,
        'curr_i': curr_i,
        'filled_i': filled_i,
        't': t,
        'ep': ep
    }


def cleanupSharedMemory(shared_memory):
    """
    Cleanup shared memory blocks.
    """
    for block in shared_memory.values():
        block.close()
        block.unlink()


def _selected_config(config=None):
    selected = config if config is not None else getattr(Config, 'config', None)
    if selected is None:
        raise ValueError(
            "Set config = (num_gpu_processes, num_cpu_processes, cores_per_cpu_process) in Config.py."
        )

    try:
        num_gpu_processes, num_cpu_processes, cores_per_cpu_process = selected
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Config.config must be a 3-item tuple: "
            "(num_gpu_processes, num_cpu_processes, cores_per_cpu_process)."
        ) from exc

    current_config = (
        int(num_gpu_processes),
        int(num_cpu_processes),
        int(cores_per_cpu_process),
    )

    num_gpu_processes, num_cpu_processes, cores_per_cpu_process = current_config
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
            f"Config.config requests {num_gpu_processes} GPU process(es), "
            f"but only {torch.cuda.device_count()} CUDA device(s) are visible."
        )

    return current_config


class HeteroMARLRunner:
    def launch(self, MARL_train, shared_memory_params, config=None):
        current_config = _selected_config(config)
        t = np.ndarray((1,), dtype=np.int32, buffer=shared_memory_params['t'].buf)
        current_episode = t[0] // Config.episode_length
        n_episodes = max(0, Config.n_episodes - current_episode)

        num_gpu_processes, num_cpu_processes, _ = current_config
        world_size = num_gpu_processes + num_cpu_processes

        print("Hetero_MARL configuration:", current_config)
        print("Training episodes remaining:", n_episodes)

        ctx = mp.get_context('fork')
        processes = [
            ctx.Process(
                target=MARL_train,
                args=(rank, world_size, shared_memory_params, n_episodes, current_config)
            )
            for rank in range(world_size)
        ]

        for process in processes:
            process.start()

        for process in processes:
            process.join()

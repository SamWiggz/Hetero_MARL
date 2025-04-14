import optuna
import psutil
import math
import time
import numpy as np

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import multiprocessing.shared_memory as shm
from multiprocessing import Process, Queue
import sys
import os
from shared_buffer import ReplayBuffer
from gym.spaces import Box

sys.path.append('../')

from utils import make_parallel_env

import Config

def initSharedMemory():
    """
    Initialize Shared Memory Blocks and Inter-process Parameters

    Parameters:
        obs_shm (shared_memory): Shared Memory Block for Observations
        ac_shm (shared_memory): Shared Memory Block for Actions
        rew_shm (shared_memory): Shared Memory Block for Rewards
        next_obs_shm (shared_memory): Shared Memory Block for Next Observations
        done_shm (shared_memory): Shared Memory Block for Done Signals

        curr_i (shared_memory): Shared Memory Block for Current Index
        filled_i (shared_memory): Shared Memory Block for Filled Index

        t (shared_memory): Shared Memory Block for Time
        ep (shared_memory): Shared Memory Block for Episodes

    Returns:
        shared_memory_params (dict): Dictionary of Shared Memory Blocks
    """
    # Dummy env to get size observation and action dimensions
    env = make_parallel_env(Config.env_id, Config.n_rollout_threads, Config.seed,
                            Config.discrete_action)

    # Create shared arrays for Shared Replay
    obs_shm = shm.SharedMemory(create=True, size=Config.buffer_length * sum([obsp.shape[0] for obsp in env.observation_space]) * np.dtype(np.float64).itemsize)
    ac_shm = shm.SharedMemory(create=True, size=Config.buffer_length * sum([acsp.shape[0] if isinstance(acsp, Box) else acsp.n for acsp in env.action_space]) * np.dtype(np.float64).itemsize)
    rew_shm = shm.SharedMemory(create=True, size=Config.buffer_length * len(env.agent_types) * np.dtype(np.float64).itemsize)
    next_obs_shm = shm.SharedMemory(create=True, size=Config.buffer_length * sum([obsp.shape[0] for obsp in env.observation_space]) * np.dtype(np.float64).itemsize)
    done_shm = shm.SharedMemory(create=True, size=Config.buffer_length * len(env.agent_types) * np.dtype(np.float64).itemsize)

    ### Close Dummy Environment ###
    env.close()

    curr_i = shm.SharedMemory(create=True, size=np.dtype(np.int32).itemsize)
    filled_i = shm.SharedMemory(create=True, size=np.dtype(np.int32).itemsize)
    t = shm.SharedMemory(create=True, size=np.dtype(np.int32).itemsize)
    ep = shm.SharedMemory(create=True, size=np.dtype(np.int32).itemsize)

    # Initialize shared memory blocks
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
    Cleanup Shared Memory Blocks
    """
    for shm in shared_memory.values():
        shm.close()
        shm.unlink()

class MARLSystemMapper:
    def __init__(self):
        self.max_cores = psutil.cpu_count(logical=False)  # Number of cores
        self.max_cpu_processes = psutil.cpu_count(logical=False) # Number of CPU processes
        self.max_gpu_processes = torch.cuda.device_count()  # Number of accelerators


        self.total_configs = self.compute_configurations() # Number of possible configuration
        self.n_calls = math.floor(Config.percentage_search * self.total_configs) # Number of searches

        print("Total Configurations:", self.total_configs)
        print("Number of Searches:", self.n_calls)

        self.param_queue = Queue()
        self.tried_configs = set()

    def compute_configurations(self):
        """
        Compute the design space: combinations of accelerator learners, CPU learners, and cores per CPU learner.

        Parameters: None
        Returns: 
            total_configurations (int): Total number of configurations
        """
        total_configurations = 0
        for g in range(self.max_gpu_processes + 1):
            if g == 0: 
                for c in range(1, self.max_cpu_processes + 1):
                    max_cores_per_learner = self.max_cores // c
                    total_configurations += max_cores_per_learner
            else:
                total_configurations += self.max_cores
                for c in range(1, self.max_cpu_processes + 1):
                    max_cores_per_learner = self.max_cores // c
                    total_configurations += max_cores_per_learner
        
        return total_configurations

    def objective(self, MARL_train, shared_memory_params, current_config=None, trial=None):
        ### Current Config: (x,y,z) ###
        ### x: Number of Learners (GPU)
        ### y: Number of Learners (CPU)
        ### z: Number of Cores per CPU Learner

        tuner = False
        # Choose new configuration to test using Bayesian Optimization
        if trial != None:
            num_gpu_learners = trial.suggest_int("num_gpu_learners", 0, torch.cuda.device_count())

            # Adjust the range for CPU learners to avoid the (0,0) case
            if num_gpu_learners == 0:
                num_cpu_learners = trial.suggest_int("num_cpu_learners", 1, 4)  # Must have at least 1 CPU learner
            else:
                num_cpu_learners = trial.suggest_int("num_cpu_learners", 0, 4)  # GPU present, so CPU can be 0

            # If num_cpu_learners > 0, suggest cores per learner
            if num_cpu_learners > 0:
                min_cores_per_learner = 1
                max_cores_per_learner = psutil.cpu_count(logical=False) // num_cpu_learners
                cores_per_cpu_learner = trial.suggest_int(
                    "cores_per_cpu_learner", min_cores_per_learner, max_cores_per_learner
                )
            else:
                cores_per_cpu_learner = trial.suggest_int("cores_per_cpu_learner", 1, psutil.cpu_count(logical=False))  # No CPU learners
            current_config = (num_gpu_learners, num_cpu_learners, cores_per_cpu_learner)
            n_episodes = 1
            tuner = True
            if current_config in self.tried_configs:
                raise optuna.exceptions.TrialPruned() 
            else:
                self.tried_configs.add(current_config)
        ### Warmup Iterations ###
        elif current_config == None:
            n_episodes = math.ceil(Config.batch_size / Config.episode_length)
            current_config = (1,0, psutil.cpu_count(logical=False))
            num_gpu_learners, num_cpu_learners, cores_per_cpu_learner = current_config
        ### Optimal Config Iterations ###
        else:
            t = np.ndarray((1,), dtype=np.int32, buffer=shared_memory_params['t'].buf)
            current_episode = t[0] // Config.episode_length
            n_episodes = Config.n_episodes - current_episode # Remaining Episodes
            num_gpu_learners, num_cpu_learners, cores_per_cpu_learner = current_config

        print("Current Configuration: ", current_config)
        world_size = num_gpu_learners + num_cpu_learners
        ctx = mp.get_context('fork')
        proc = [ctx.Process(target = MARL_train,
                args=(i, world_size, shared_memory_params, n_episodes, current_config, self.param_queue, tuner))
                for i in range(world_size)]

        for p in proc:
            p.start()

        for p in proc:
            p.join() 

        result = self.param_queue.get()
        print("Test Iteration Time:", result)

        return result  # Optimize some performance metric

    def launch(self, MARL_train, shared_memory_params, config = None):
        ### Warmup ###
        self.objective(MARL_train, shared_memory_params)

        ### Optimize ###
        self.study = optuna.create_study(study_name='SMART', direction='minimize', sampler=optuna.samplers.GPSampler(n_startup_trials=self.n_calls))
        self.study.optimize(
            lambda trial: self.objective(MARL_train, shared_memory_params, trial = trial), 
            n_trials=self.n_calls)
            
        config_opt = self.study.best_params
        config = (config_opt['num_gpu_learners'], config_opt['num_cpu_learners'], config_opt['cores_per_cpu_learner'])

        ### Train with Optimal Parameters ###
        self.objective(MARL_train, shared_memory_params, current_config = config)
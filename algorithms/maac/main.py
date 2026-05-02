import argparse
import torch
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from shared_buffer import ReplayBuffer
from attention_sac import AttentionSAC
import Config
import math

import sys
sys.path.append('../')
from utils import make_parallel_env, is_cpu_process, gpu_mapping, get_device, get_env_config

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
import time

from balancer import Runtime_Load_Balancer
from hetero_marl import initSharedMemory, cleanupSharedMemory, HeteroMARLRunner

def MARL_train(rank, world_size, shared_memory_params, n_episodes, current_config):

    num_gpu_processes, num_cpu_processes, cores_per_cpu_process = current_config
    if num_cpu_processes == 0:
        torch.cuda.set_device(get_device(rank, num_cpu_processes))
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
    else :
        dist.init_process_group('gloo', rank=rank, world_size=world_size)

    torch.manual_seed(Config.seed)
    torch.cuda.manual_seed_all(Config.seed)
    np.random.seed(Config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    t = np.ndarray((1,), dtype=np.int32, buffer=shared_memory_params['t'].buf)
    ep = np.ndarray((1,), dtype=np.int32, buffer=shared_memory_params['ep'].buf)

    # set training threads
    if get_device(rank, num_cpu_processes) == torch.device('cpu'):
        torch.set_num_threads(cores_per_cpu_process)
    elif rank != 0:
        torch.set_num_threads(1)
    else: # if GPU thread is the actor thread
        torch.set_num_threads(1)

    # only rank 0 does data collection
    env = make_parallel_env(Config.env_id, Config.n_rollout_threads, Config.seed,
                            Config.discrete_action, get_env_config(Config))

    model = AttentionSAC.init_from_env(env,
                                       tau=Config.tau,
                                       pi_lr=Config.pi_lr,
                                       q_lr=Config.q_lr,
                                       gamma=Config.gamma,
                                       pol_hidden_dim=Config.pol_hidden_dim,
                                       critic_hidden_dim=Config.critic_hidden_dim,
                                       attend_heads=Config.attend_heads,
                                       reward_scale=Config.reward_scale,
                                       device = get_device(rank, num_cpu_processes),
                                       rank = rank,
                                       world_size = world_size)

    replay_buffer = ReplayBuffer(Config.buffer_length, len(env.agent_types),
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space],
                                  shared_memory_params['obs_shm'], shared_memory_params['ac_shm'], shared_memory_params['rew_shm'],
                                  shared_memory_params['next_obs_shm'], shared_memory_params['done_shm'], shared_memory_params['curr_i'], shared_memory_params['filled_i'])

    if rank != 0:
        env.close()

    balancer = Runtime_Load_Balancer(Config.batch_size, get_device(rank, num_cpu_processes) == torch.device('cpu'), rank, world_size, num_cpu_processes, num_gpu_processes)

    # Start Training
    dist.barrier()
    train_start = time.perf_counter()
    dc_tot = 0
    mu_tot = 0
    dc_count = 0
    mu_count = 0
    timing_log_interval_updates = max(0, int(getattr(Config, 'timing_log_interval_updates', 10)))
    next_timing_log_update = timing_log_interval_updates
    current_episode = t[0] // Config.episode_length
    profile = 0
    for ep_i in range(current_episode, current_episode + n_episodes, Config.n_rollout_threads):
        if rank == 0:
            ep[0] = ep_i + Config.n_rollout_threads
            print("Episodes %i-%i of %i" % (ep_i + 1,
                                    min(ep_i + Config.n_rollout_threads, Config.n_episodes),
                                    Config.n_episodes))
            obs = env.reset()
            # obs.shape = (n_rollout_threads, nagent)(nobs), nobs differs per agent so not tensor

        for et_i in range(Config.episode_length):
            ###
            # Data Collection
            ###
            dist.barrier()
            dc_clock = time.perf_counter()
            if rank == 0:
                # rearrange observations to be per agent, and convert to torch Variable
                torch_obs = [torch.stack([torch.tensor(arr[i], dtype=torch.float32, requires_grad=False) for arr in obs]) for i in range(model.nagents)]
                # get actions as torch Variables
                with torch.no_grad():
                    torch_agent_actions = model.step(torch_obs, explore=True)
                # convert actions to numpy arrays
                agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
                # rearrange actions to be per environment
                actions = [[ac[i] for ac in agent_actions] for i in range(Config.n_rollout_threads)]
                next_obs, rewards, dones, infos = env.step(actions)
                replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
                obs = next_obs

            if rank == 0:
                t[0] += Config.n_rollout_threads

            dist.barrier()
            dc_tot += time.perf_counter() - dc_clock

            ###
            # Model Update
            ###
            if (len(replay_buffer) >= Config.batch_size and
                (t[0] % Config.steps_per_update) < Config.n_rollout_threads):
                    dc_count += 1
                    dist.barrier()
                    mu_start = time.perf_counter()
                    for u_i in range(Config.n_updates):
                        sample = replay_buffer.sample(balancer.get_subbatch_size(),
                                                device=get_device(rank, num_cpu_processes))
                        profile += model.update_critic(sample, logger=None)
                        profile += model.update_policies(sample, logger=None)
                        model.update_all_targets()
                    dist.barrier()

                    mu_tot += time.perf_counter() - mu_start
                    mu_last = time.perf_counter() - mu_start
                    mu_count += 1
                    balancer.update(mu_last, profile)
                    profile = 0

                    if rank == 0:
                        if next_timing_log_update and mu_count >= next_timing_log_update:
                            print(
                                "Avg Data Collection Time: {:.4f}s | "
                                "Avg Model Update Time: {:.4f}s".format(
                                    dc_tot / dc_count if dc_count else 0.0,
                                    mu_tot / mu_count if mu_count else 0.0,
                                )
                            )
                            next_timing_log_update += timing_log_interval_updates
                        model.prep_rollouts()

    if rank == 0:
        env.close()
        print(f"Total Train Time: {time.perf_counter() - train_start:.4f}s")
        print(f"Avg Data Collection Time: {(dc_tot / dc_count if dc_count else 0.0):.4f}s")
        print(f"Avg Model Update Time: {(mu_tot / mu_count if mu_count else 0.0):.4f}s")
        print("Config: ", current_config)
        print("Environment: ", Config.env_id)
        print("Agents: ", model.nagents)
        print("Batch Size: ", Config.batch_size)

    dist.barrier()
    dist.destroy_process_group()

if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = Config.port

    ### Create Shared Memory ###
    shared_memory_params = initSharedMemory()

    runner = HeteroMARLRunner()
    runner.launch(MARL_train, shared_memory_params)

    ### Cleanup Shared Memory ###
    cleanupSharedMemory(shared_memory_params)

import numpy as np
from torch.autograd import Variable
from torch import Tensor
import multiprocessing.shared_memory as shm
import torch
import Config

class ReplayBuffer(object):
    """
    Replay Buffer for multi-agent RL with parallel rollouts using shared memory
    """
    def __init__(self, max_steps, num_agents, obs_dims, ac_dims,obs_shm, ac_shm, rew_shm, next_obs_shm, done_shm, curr_i, filled_i):
        """
        Inputs:
            max_steps (int): Maximum number of timepoints to store in buffer
            num_agents (int): Number of agents in environment
            obs_dims (list of ints): number of observation dimensions for each agent
            ac_dims (list of ints): number of action dimensions for each agent
        """
        self.max_steps = max_steps
        self.num_agents = num_agents
        self.filled_i = np.ndarray((1,), dtype=np.int32, buffer=filled_i.buf) # index of first empty location in buffer (last index when full)
        self.curr_i = np.ndarray((1,), dtype=np.int32, buffer=curr_i.buf)  # current index to write to (overwrite oldest data)


        # Create shared memory for each buffer
        self.obs_shm = obs_shm
        self.ac_shm = ac_shm
        self.rew_shm = rew_shm
        self.next_obs_shm = next_obs_shm
        self.done_shm = done_shm

        # Create numpy arrays backed by shared memory
        self.obs_buffs = [np.ndarray((max_steps, odim), dtype=np.float32, buffer=self.obs_shm.buf) for odim in obs_dims]
        self.ac_buffs = [np.ndarray((max_steps, adim), dtype=np.float32, buffer=self.ac_shm.buf) for adim in ac_dims]
        self.rew_buffs = np.ndarray((max_steps, num_agents), dtype=np.float32, buffer=self.rew_shm.buf)
        self.next_obs_buffs = [np.ndarray((max_steps, odim), dtype=np.float32, buffer=self.next_obs_shm.buf) for odim in obs_dims]
        self.done_buffs = np.ndarray((max_steps, num_agents), dtype=np.float32, buffer=self.done_shm.buf)

        np.random.seed(Config.seed)

    def __len__(self):
        return self.filled_i[0]

    def push(self, obs, actions, rewards, next_obs, dones):
        observations = np.array(obs, dtype=object)
        next_observations = np.array(next_obs, dtype=object)
        nentries = observations.shape[0]  # handle multiple parallel environments
        #print(rewards)
        if self.curr_i[0] + nentries > self.max_steps:
            rollover = self.max_steps - self.curr_i[0]  # num of indices to roll over
            for agent_i in range(self.num_agents):
                self.obs_buffs[agent_i] = np.roll(self.obs_buffs[agent_i], rollover, axis=0)
                self.ac_buffs[agent_i] = np.roll(self.ac_buffs[agent_i], rollover, axis=0)
                self.rew_buffs[:, agent_i] = np.roll(self.rew_buffs[:, agent_i], rollover)
                self.next_obs_buffs[agent_i] = np.roll(self.next_obs_buffs[agent_i], rollover, axis=0)
                self.done_buffs[:, agent_i] = np.roll(self.done_buffs[:, agent_i], rollover)
            self.curr_i[0] = 0
            self.filled_i[0] = self.max_steps
        for agent_i in range(self.num_agents):
            self.obs_buffs[agent_i][self.curr_i[0]:self.curr_i[0] + nentries] = np.vstack(observations[:, agent_i])
            self.ac_buffs[agent_i][self.curr_i[0]:self.curr_i[0] + nentries] = actions[agent_i]
            self.rew_buffs[self.curr_i[0]:self.curr_i[0] + nentries, agent_i] = rewards[:, agent_i]
            self.next_obs_buffs[agent_i][self.curr_i[0]:self.curr_i[0] + nentries] = np.vstack(next_observations[:, agent_i])
            self.done_buffs[self.curr_i[0]:self.curr_i[0] + nentries, agent_i] = dones[:, agent_i]
        self.curr_i[0] += nentries
        if self.filled_i[0] < self.max_steps:
            self.filled_i[0] += nentries
        if self.curr_i[0] == self.max_steps:
            self.curr_i[0] = 0

    def sample(self, a_i, N, device='cpu', norm_rews=False, rank = 0):
        np.random.seed((rank+1) * 100)
        inds = np.random.choice(np.arange(self.filled_i[0]), size=N, replace=False)
        cast = lambda x: Variable(Tensor(x), requires_grad=False).to(device)
        if norm_rews:
            ret_rews = [cast((self.rew_buffs[i][inds] -
                              self.rew_buffs[i][:self.filled_i].mean()) /
                             self.rew_buffs[i][:self.filled_i].std())
                        for i in range(self.num_agents)]
        else:
            ret_rews = cast(self.rew_buffs[inds,a_i])
        return (cast(self.obs_buffs[a_i][inds]),
                cast(self.ac_buffs[a_i][inds]),
                ret_rews,
                cast(self.next_obs_buffs[a_i][inds]),
                cast(self.done_buffs[inds,a_i]))

    def get_average_rewards(self, N):
        if self.filled_i[0] == self.max_steps:
            inds = np.arange(self.curr_i[0] - N, self.curr_i[0])  # allow for negative indexing
        else:
            inds = np.arange(max(0, self.curr_i[0] - N), self.curr_i[0])
        return [self.rew_buffs[inds, i].mean() for i in range(self.num_agents)]

    def close(self):
        # Close and unlink shared memory
        self.obs_shm.close()
        self.ac_shm.close()
        self.rew_shm.close()
        self.next_obs_shm.close()
        self.done_shm.close()

    def unlink(self):
        # Unlink shared memory
        self.obs_shm.unlink()
        self.ac_shm.unlink()
        self.rew_shm.unlink()
        self.next_obs_shm.unlink()
        self.done_shm.unlink()

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

        fin_dc (shared_memory): Shared Memory Block for Data Collection Completion Signal
        fin_mu (shared_memory): Shared Memory Block for Model Update Completion Signal
        fin_done (shared_memory): Shared Memory Block for Completion Signal
        t (shared_memory): Shared Memory Block for Time
        ep (shared_memory): Shared Memory Block for Episodes

    Returns:
        shared_memory_params (dict): Dictionary of Shared Memory Blocks
    """
    ### Initialize Dummy Environment to setup Replay Buffer ###
    grid_config = GridConfig(num_agents=Config.n_agents,  # number of agents
                size=Config.mapsize, # size of the grid
                density=0.4,  # obstacle density
                seed=1,  # set to None for random
                        # obstacles, agents and targets
                        # positions at each reset
                max_episode_steps=Config.episode_length,  # horizon
                obs_radius=3,  # defines field of view
                )
    env = pogema_v0(grid_config=grid_config)

    # Create shared memory blocks with unique names
    obs_shm = shm.SharedMemory(create=True, size=Config.buffer_length * sum(env.observation_space.shape[0]*env.observation_space.shape[1]*env.observation_space.shape[2] for i in range(Config.n_agents)) * np.dtype(np.float64).itemsize)
    ac_shm = shm.SharedMemory(create=True, size=Config.buffer_length * sum(env.action_space.shape[0] if isinstance(env.action_space, Box) else env.action_space.n for i in range(Config.n_agents)) * np.dtype(np.float64).itemsize)
    rew_shm = shm.SharedMemory(create=True, size=Config.buffer_length * Config.n_agents * np.dtype(np.float64).itemsize)
    next_obs_shm = shm.SharedMemory(create=True, size=Config.buffer_length * sum(env.observation_space.shape[0]*env.observation_space.shape[1]*env.observation_space.shape[2] for i in range(Config.n_agents)) * np.dtype(np.float64).itemsize)
    done_shm = shm.SharedMemory(create=True, size=Config.buffer_length * Config.n_agents * np.dtype(np.float64).itemsize)

    ### Close Dummy Environment ###
    env.close()

    curr_i = shm.SharedMemory(create=True, size=np.dtype(np.int32).itemsize)
    filled_i = shm.SharedMemory(create=True, size=np.dtype(np.int32).itemsize)
    # fin_dc = shm.SharedMemory(create=True, size=np.dtype(np.int32).itemsize)
    # fin_mu = shm.SharedMemory(create=True, size=np.dtype(np.int32).itemsize)
    # fin_done = shm.SharedMemory(create=True, size=np.dtype(np.int32).itemsize)
    # t = shm.SharedMemory(create=True, size=np.dtype(np.int32).itemsize)
    # ep = shm.SharedMemory(create=True, size=np.dtype(np.int32).itemsize)

    # Initialize shared memory blocks
    np.ndarray((1,), dtype=np.int32, buffer=curr_i.buf)[0] = 0
    np.ndarray((1,), dtype=np.int32, buffer=filled_i.buf)[0] = 0
    # np.ndarray((1,), dtype=np.int32, buffer=fin_dc.buf)[0] = 0
    # np.ndarray((1,), dtype=np.int32, buffer=fin_mu.buf)[0] = 0
    # np.ndarray((1,), dtype=np.int32, buffer=fin_done.buf)[0] = 0
    # np.ndarray((1,), dtype=np.int32, buffer=t.buf)[0] = 0
    # np.ndarray((1,), dtype=np.int32, buffer=ep.buf)[0] = 0

    return {
        'obs_shm': obs_shm,
        'ac_shm': ac_shm,
        'rew_shm': rew_shm,
        'next_obs_shm': next_obs_shm,
        'done_shm': done_shm,
        'curr_i': curr_i,
        'filled_i': filled_i
        # 'fin_dc': fin_dc,
        # 'fin_mu': fin_mu,
        # 'fin_done': fin_done,
        # 't': t,
        # 'ep': ep
    }

def cleanupSharedMemory(shared_memory):
    """
    Cleanup Shared Memory Blocks
    """
    for shm in shared_memory.values():
        shm.close()
        shm.unlink()
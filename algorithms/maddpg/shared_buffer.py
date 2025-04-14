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

    def sample(self, N, device='cpu', norm_rews=True, rank = 0):
        np.random.seed((rank+1) * 100)
        #print(device, self.filled_i[0])
        #print(device, self.curr_i[0])
        inds = np.random.choice(np.arange(self.filled_i[0]), size=N, replace=False)
        cast = lambda x: Variable(Tensor(x), requires_grad=False).to(device)
        #print(device, inds)
        #print(device, cast)
        if norm_rews:
            ret_rews = [cast((self.rew_buffs[inds, i] - self.rew_buffs[:self.filled_i[0], i].mean()) /
                             self.rew_buffs[:self.filled_i[0], i].std()) for i in range(self.num_agents)]
        else:
            ret_rews = [cast(self.rew_buffs[inds, i]) for i in range(self.num_agents)]
        #print(device, ret_rews)
        return ([cast(self.obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
                ret_rews,
                [cast(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.done_buffs[inds, i]) for i in range(self.num_agents)])

    def get_average_rewards(self, N):
        if self.filled_i[0] == self.max_steps:
            inds = np.arange(self.curr_i[0] - N, self.curr_i[0])  # allow for negative indexing
        else:
            inds = np.arange(max(0, self.curr_i[0] - N), self.curr_i[0])
        return [self.rew_buffs[inds, i].mean() for i in range(self.num_agents)]
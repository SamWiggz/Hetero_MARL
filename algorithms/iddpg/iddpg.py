import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from gym.spaces import Box
from torch.autograd import Variable

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist

import sys
import os
sys.path.append('../')

from utils import hard_update, soft_update, gumbel_softmax, onehot_from_logits, OUNoise
import Config
import time

from torch.futures import Future

class AllReduceTimer:
    def __init__(self, rank):
        self.rank = rank
        self.times = []

    def reset(self):
        self.times.clear()

    def hook(self, state, bucket):
        """
        Communication hook for DDP.
        - `state` is unused.
        - `bucket` contains the gradient tensors to be all-reduced.
        """
        start_time = time.perf_counter()

        # Start async all_reduce and get PyTorch Future
        fut = dist.all_reduce(bucket.buffer(), async_op=True).get_future()

        # Properly return a Future[Tensor] by chaining the hook
        def after_all_reduce(fut: Future):
            end_time = time.perf_counter()
            elapsed_ms = (end_time - start_time)
            self.times.append(elapsed_ms)

            # ✅ Return the Tensor, not a list or other wrapper
            return bucket.buffer()

        # ✅ Return Future[Tensor]
        return fut.then(after_all_reduce)

    def get_total_time(self):
        return sum(self.times)


    def accumulated_total(self):
        return self.get_total_time()

    def print_summary(self, prefix=""):
        total = self.get_total_time()
        print(f"[Rank {self.rank}] {prefix} AllReduce total time: {total:.3f} ms")

class PolicyNN(nn.Module):
    def __init__(self, num_in_pol, num_out_pol):
        super(PolicyNN, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(num_in_pol),
            nn.Linear(num_in_pol, Config.hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(Config.hidden_sizes[1], Config.hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(Config.hidden_sizes[1], num_out_pol)
        )
        # Initialize the weights and bias of BatchNorm1d
        self.model[0].weight.data.fill_(1)
        self.model[0].bias.data.fill_(0)

    def forward(self, state):
        return self.model(state)

class CriticNN(nn.Module):
    def __init__(self, num_in_critic):
        super(CriticNN, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(num_in_critic),
            nn.Linear(num_in_critic, Config.hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(Config.hidden_sizes[1], Config.hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(Config.hidden_sizes[1], 1)
        )

        # Initialize the weights and bias of BatchNorm1d
        self.model[0].weight.data.fill_(1)
        self.model[0].bias.data.fill_(0)

    def forward(self, state_action):
        return self.model(state_action)

class DDPGAgent(object):
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, discrete_action=True, device = 'cpu', world_size = 1, rank = 0):
        self.policy_nn = PolicyNN(num_in_pol, num_out_pol).to(device)
        self.target_policy_nn = PolicyNN(num_in_pol, num_out_pol).to(device)
        self.critic_nn = CriticNN(num_in_critic).to(device)
        self.target_critic_nn = CriticNN(num_in_critic).to(device)

        hard_update(self.target_policy_nn, self.policy_nn)
        hard_update(self.target_critic_nn, self.critic_nn)

        if rank == 0:
            self.policy_nn_inf = PolicyNN(num_in_pol, num_out_pol).to('cpu')
            hard_update(self.policy_nn_inf, self.policy_nn)


        if world_size > 1:
            self.policy_nn = DDP(self.policy_nn)
            self.target_policy_nn = DDP(self.target_policy_nn)
            self.critic_nn = DDP(self.critic_nn)
            self.target_critic_nn = DDP(self.target_critic_nn)

            self.policy_timer = AllReduceTimer(rank)
            self.critic_timer = AllReduceTimer(rank)

            self.policy_nn.register_comm_hook(state=None, hook=self.policy_timer.hook)
            self.critic_nn.register_comm_hook(state=None, hook=self.critic_timer.hook)

        self.policy_optim = torch.optim.Adam(params=self.policy_nn.parameters(), lr = Config.lr)
        self.critic_optim = torch.optim.Adam(params=self.critic_nn.parameters(), lr = Config.lr)

        if not discrete_action:
            self.exploration = OUNoise(num_out_pol)
        else:
            self.exploration = 0.3  # epsilon for eps-greedy
        self.discrete_action = discrete_action

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        action = self.policy_nn_inf(obs)
        if self.discrete_action:
            if explore:
                action = gumbel_softmax(action, hard=True)
            else:
                action = onehot_from_logits(action)
        else:  # continuous action
            if explore:
                action += Variable(Tensor(self.exploration.noise()),
                                   requires_grad=False)
            action = action.clamp(-1, 1)
        return action

class IDDPG(object):
    def __init__(self, agent_init_params, alg_types,
                 gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64,
                 discrete_action=False, rank=0, device = 'cpu', world_size = 1):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent (DDPG
                                       or MADDPG)
        """
        self.num_agents = len(alg_types)
        self.alg_types = alg_types
        self.agents = [DDPGAgent(discrete_action=discrete_action, device = device, world_size = world_size, rank = rank,
                                 **params)
                       for params in agent_init_params]
        self.agent_init_params = agent_init_params
        self.gamma = Config.gamma
        self.tau = Config.tau
        self.lr = Config.lr
        self.discrete_action = discrete_action
        self.niter = 0
        self.MSELoss = torch.nn.MSELoss()
        self.pol_dev = device  # device for policies
        self.critic_dev = device  # device for critics
        self.trgt_pol_dev = device  # device for target policies
        self.trgt_critic_dev = device  # device for target critics
        self.rank = rank
        self.world_size = world_size

    @property
    def policies(self):
        return [a.policy_nn for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy_nn for a in self.agents]

    def scale_noise(self, scale):
        """
        Scale noise for each agent
        Inputs:
            scale (float): scale of noise
        """
        for a in self.agents:
            a.scale_noise(scale)

    def reset_noise(self):
        for a in self.agents:
            a.reset_noise()

    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        return [a.step(obs, explore=explore) for a, obs in zip(self.agents,
                                                                 observations)]

    def update(self, sample, agent_i, device, logger=None):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
        """

        #print(sample)
        obs, acs, rews, next_obs, dones = sample
        curr_agent = self.agents[agent_i]

        ##
        ## Update Critic
        ##
        # FW Target Policy
        if self.discrete_action:
            trgt_vf_in = torch.cat((next_obs,
                                    onehot_from_logits(
                                        curr_agent.target_policy_nn(
                                            next_obs))),
                                dim=1)
        else:
            trgt_vf_in = torch.cat((next_obs,
                                    curr_agent.target_policy_nn(next_obs)),
                                dim=1)
        target_value = (rews.view(-1, 1) + self.gamma *
                        curr_agent.target_critic_nn(trgt_vf_in) *
                        (1 - dones.view(-1, 1)))

        vf_in = torch.cat((obs, acs), dim=1)
        actual_value = curr_agent.critic_nn(vf_in)
        vf_loss = self.MSELoss(actual_value, target_value.detach())

        curr_agent.critic_optim.zero_grad()
        vf_loss.backward()
        torch.nn.utils.clip_grad_norm_(curr_agent.critic_nn.parameters(), 0.5)
        curr_agent.critic_optim.step()

        ##
        ## Update Policy
        ##
        if self.discrete_action:
            curr_pol_out = curr_agent.policy_nn(obs)
            curr_pol_vf_in = gumbel_softmax(curr_pol_out, device=device, hard=True)
        else:
            curr_pol_out = curr_agent.policy_nn(obs)
            curr_pol_vf_in = curr_pol_out
        vf_in = torch.cat((obs, curr_pol_vf_in),
                            dim=1)

        pol_loss = -curr_agent.critic_nn(vf_in).mean()
        pol_loss += (curr_pol_out**2).mean() * 1e-3

        curr_agent.policy_optim.zero_grad()
        #print(device, "here2")
        pol_loss.backward()
        #print(device, "here3")
        torch.nn.utils.clip_grad_norm_(curr_agent.policy_nn.parameters(), 0.5)
        curr_agent.policy_optim.step()

        # if logger is not None:
        #     logger.add_scalars('agent%i/losses' % agent_i,
        #                        {'vf_loss': vf_loss,
        #                         'pol_loss': pol_loss},
        #                        self.niter)

        if self.world_size > 1:
            profile =  curr_agent.policy_timer.get_total_time() + curr_agent.critic_timer.get_total_time()
            curr_agent.policy_timer.reset()
            curr_agent.critic_timer.reset()
        else:
            profile = 0
        return profile


    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        with torch.no_grad():
            for a in self.agents:
                soft_update(a.target_critic_nn, a.critic_nn, self.tau)
                soft_update(a.target_policy_nn, a.policy_nn, self.tau)
            self.niter += 1

    def prep_training(self, device='cpu'):
        for a in self.agents:
            a.policy_nn.train()
            a.critic_nn.train()
            a.target_policy_nn.train()
            a.target_critic_nn.train()
        fn = lambda x: x.to(device)
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy_nn = fn(a.policy_nn)
            self.pol_dev = device
        if not self.critic_dev == device:
            for a in self.agents:
                a.critic_nn = fn(a.critic_nn)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy_nn = fn(a.target_policy_nn)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            for a in self.agents:
                a.target_critic_nn = fn(a.target_critic_nn)
            self.trgt_critic_dev = device

    def prep_rollouts(self):
        for a in self.agents:
            if self.world_size > 1:
                hard_update(a.policy_nn_inf, a.policy_nn.module)
            else:
                hard_update(a.policy_nn_inf, a.policy_nn)
            a.policy_nn_inf.eval()

    @classmethod
    def init_from_env(cls, env, agent_alg="DDPG", adversary_alg="DDPG",
                      gamma=0.95, tau=0.01, lr=0.01, hidden_dim=128, rank = 0, device = 'cpu', world_size = 1):
        """
        Instantiate instance of this class from multi-agent environment
        """
        agent_init_params = []
        alg_types = [adversary_alg if atype == 'adversary' else agent_alg for
                     atype in env.agent_types]
        for acsp, obsp, algtype in zip(env.action_space, env.observation_space,
                                       alg_types):
            num_in_pol = obsp.shape[0]
            if isinstance(acsp, Box):
                discrete_action = False
                get_shape = lambda x: x.shape[0]
            else:  # Discrete
                discrete_action = True
                get_shape = lambda x: x.n
            num_out_pol = get_shape(acsp)
            num_in_critic = obsp.shape[0] + get_shape(acsp)
            agent_init_params.append({'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol,
                                      'num_in_critic': num_in_critic})
        init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr,
                     'hidden_dim': hidden_dim,
                     'alg_types': alg_types,
                     'agent_init_params': agent_init_params,
                     'discrete_action': discrete_action,
                     'rank': rank,
                     'device': device,
                     'world_size': world_size}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

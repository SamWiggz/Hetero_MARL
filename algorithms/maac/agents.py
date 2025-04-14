from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
import sys
sys.path.append('../')
from utils import hard_update, gumbel_softmax, onehot_from_logits
from policies import DiscretePolicy

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist

from torch.futures import Future
import time

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

class AttentionAgent(object):
    """
    General class for Attention agents (policy, target policy)
    """
    def __init__(self, num_in_pol, num_out_pol, hidden_dim=64,
                 lr=0.01, onehot_dim=0, device = 'cpu', world_size = 1, rank = 0):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
        """
        self.policy = DiscretePolicy(num_in_pol, num_out_pol,
                                     hidden_dim=hidden_dim,
                                     onehot_dim=onehot_dim).to(device)
        self.target_policy = DiscretePolicy(num_in_pol,
                                            num_out_pol,
                                            hidden_dim=hidden_dim,
                                            onehot_dim=onehot_dim).to(device)
        
        hard_update(self.target_policy, self.policy)
        
        if rank == 0:
            self.policy_inf = DiscretePolicy(num_in_pol, num_out_pol,
                                     hidden_dim=hidden_dim,
                                     onehot_dim=onehot_dim).to('cpu')
            hard_update(self.policy_inf, self.policy)

        if world_size > 1:
            self.policy = DDP(self.policy)
            self.target_policy = DDP(self.target_policy)

            self.policy_timer = AllReduceTimer(rank)
            self.policy.register_comm_hook(state=None, hook=self.policy_timer.hook)

        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)

    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to sample
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        return self.policy_inf(obs, sample=explore)

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])

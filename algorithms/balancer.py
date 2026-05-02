import psutil
import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ReduceOp
import time

class Runtime_Load_Balancer:
    def __init__(self, batch_size, is_cpu, rank, world_size, num_cpu_processes = 1, num_gpu_processes = 0, cpu_gpu_ratio=0.5) :
        self.is_cpu = is_cpu
        self.num_cpu = num_cpu_processes
        self.num_gpu = num_gpu_processes

        self.batch_size = batch_size
        self.cpu_gpu_ratio = cpu_gpu_ratio
        self.rank = rank
        self.world_size = world_size

    def get_subbatch_size(self):
        if self.num_cpu == 0:
            self.cpu_gpu_ratio = 0
        if self.num_gpu == 0:
            self.cpu_gpu_ratio = 1
        cpu_batch_size = int(self.batch_size * self.cpu_gpu_ratio)
        if self.is_cpu:
            return cpu_batch_size // self.num_cpu + \
                (cpu_batch_size % self.num_cpu if self.rank == self.num_cpu - 1 else 0)
        else:
            return (self.batch_size - cpu_batch_size) // self.num_gpu+ \
                ((self.batch_size - cpu_batch_size) % self.num_gpu if self.rank == self.world_size - 1 else 0)

    def update(self, total, all_reduce):
        if self.num_cpu != 0 and self.num_gpu != 0:
            profile_time = total
            cpu_sync_time = 0
            gpu_sync_time = 0
            if self.is_cpu:
                cpu_sync_time = all_reduce
            else:
                gpu_sync_time = all_reduce

            runtime = torch.tensor([cpu_sync_time, gpu_sync_time], dtype=torch.float32)
            dist.all_reduce(runtime, op=ReduceOp.SUM)
            cpu_sync_time, gpu_sync_time = runtime.tolist()

            cpu_sync_time /= self.num_cpu
            gpu_sync_time /= self.num_gpu

            time_cpu = ((profile_time - cpu_sync_time))
            time_gpu = ((profile_time - gpu_sync_time))

            if abs(time_cpu - time_gpu) / max(time_cpu, time_gpu) \
                > 0.05:
                gpu_estimate_time = time_gpu / (1-self.cpu_gpu_ratio)
                cpu_estimate_time = time_cpu / self.cpu_gpu_ratio
                ratio = gpu_estimate_time / (cpu_estimate_time + gpu_estimate_time)
                self.cpu_gpu_ratio = min(max(0.05, round(ratio, 3)), 0.95)
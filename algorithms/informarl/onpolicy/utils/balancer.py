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
        self.cpu_gpu_ratio = 0.5
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

    def update(self, prof):
        # Load Balancer
        if prof != None and self.world_size > 1 and self.num_cpu != 0 and self.num_gpu != 0:
            profile_time = 0
            cpu_sync_time = 0
            gpu_sync_time = 0
            cuda_stream_sync_time = 0
            time_cpu = 0
            time_gpu = 0

            sort = time.perf_counter()
            #event_list = [e for e in event_list if e.key in {'ProfilerStep', 'gloo:all_reduce', 'cudaStreamSynchronize'}]
            #print(event_list)

            event_list = prof.key_averages()
            #print(event_list)
            for event in event_list:
                if 'ProfilerStep' in event.key:
                    if event.cpu_time_total > 0 :
                        profile_time = event.cpu_time_total
                if 'gloo:all_reduce' in event.key:
                    if event.cpu_time_total > 0 :
                        if self.is_cpu:
                            cpu_sync_time = event.cpu_time_total
                        else:
                            #print(event.key)
                            gpu_sync_time = event.cpu_time_total
                            #print("GPU sync time:", gpu_sync_time)
                if 'cudaStreamSynchronize' in event.key:
                    cuda_stream_sync_time = event.cpu_time_total
            if not self.is_cpu:
                gpu_sync_time -= cuda_stream_sync_time  # actual GPU sync time
            #print("balancer sort", time.perf_counter()-sort)

            #print(self.rank, profile_time, cpu_sync_time, gpu_sync_time)
            runtime = torch.tensor([cpu_sync_time, gpu_sync_time], dtype=torch.float32)
            dist.all_reduce(runtime, op=ReduceOp.SUM)
            cpu_sync_time, gpu_sync_time = runtime.tolist()
            #print("Reduced:", cpu_sync_time, gpu_sync_time)
            cpu_sync_time /= self.num_cpu
            gpu_sync_time /= self.num_gpu
            #print(profile_time, cpu_sync_time, gpu_sync_time)
            time_cpu = ((profile_time - cpu_sync_time) / 1e6)
            time_gpu = ((profile_time - gpu_sync_time) / 1e6)

            #print(self.rank, 'CPU runtime: ', time_cpu , 'GPU runtime: ', time_gpu)

            if abs(time_cpu - time_gpu) / max(time_cpu, time_gpu) \
                > 0.01:
                gpu_estimate_time = time_gpu / (1-self.cpu_gpu_ratio)
                cpu_estimate_time = time_cpu / self.cpu_gpu_ratio
                ratio = gpu_estimate_time / (cpu_estimate_time + gpu_estimate_time)
                self.cpu_gpu_ratio = min(max(0.01, round(ratio, 3)), 0.95)

            print(self.rank, 'CPU runtime: ', time_cpu , 'GPU runtime: ', time_gpu)
            print(self.rank, 'CPU_GPU Ratio:', self.cpu_gpu_ratio)
            print(self.rank, self.get_subbatch_size())

    def update2(self, total, all_reduce):
        if self.num_cpu != 0 and self.num_gpu != 0:
            #print(self.world_size)
            profile_time = total
            cpu_sync_time = 0
            gpu_sync_time = 0
            if self.is_cpu:
                cpu_sync_time = all_reduce
            else:
                gpu_sync_time = all_reduce
            #print(self.rank, profile_time, cpu_sync_time, gpu_sync_time)
            runtime = torch.tensor([cpu_sync_time, gpu_sync_time], dtype=torch.float32)
            dist.all_reduce(runtime, op=ReduceOp.SUM)
            cpu_sync_time, gpu_sync_time = runtime.tolist()
            #print("Reduced:", cpu_sync_time, gpu_sync_time)
            cpu_sync_time /= self.num_cpu
            gpu_sync_time /= self.num_gpu
            #print(profile_time, cpu_sync_time, gpu_sync_time)
            time_cpu = ((profile_time - cpu_sync_time))
            time_gpu = ((profile_time - gpu_sync_time))

            #print(self.rank, 'CPU runtime: ', time_cpu , 'GPU runtime: ', time_gpu)

            if abs(time_cpu - time_gpu) / max(time_cpu, time_gpu) \
                > 0.05:
                gpu_estimate_time = time_gpu / (1-self.cpu_gpu_ratio)
                cpu_estimate_time = time_cpu / self.cpu_gpu_ratio
                ratio = gpu_estimate_time / (cpu_estimate_time + gpu_estimate_time)
                self.cpu_gpu_ratio = min(max(0.4, round(ratio, 3)), 0.6)

                # print(self.rank, 'CPU runtime: ', time_cpu , 'GPU runtime: ', time_gpu)
                # print(self.rank, 'CPU_GPU Ratio:', self.cpu_gpu_ratio)
                # print(self.rank, self.get_subbatch_size())
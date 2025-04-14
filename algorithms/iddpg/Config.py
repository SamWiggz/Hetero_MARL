####
#### Model Definition ####
####

# IDDPG
hidden_sizes = [128, 128]
batch_size = 1024
gamma = 0.95
tau = 0.01
lr = 0.01
agent_alg = "DDPG" # "DDPG"
adversary_alg = "DDPG"
adam_eps = 1e-8

####
#### Replay Buffer ####
####
buffer_length = int(1e6)

####
#### Environment ####
####
env_id = "simple_push" #simple_speaker_listener, simple_push, multi_speaker_listener, simple_spread, simple_adversary, simple_tag, simple_crypto, fullobs_collect_treasure
n_episodes = 25000
n_rollout_threads = 32
n_training_threads = 1
n_updates = 4
steps_per_update = 100
episode_length = 25
discrete_action = True
init_noise_scale = 0.3
final_noise_scale = 0.0
seed = 1
n_exploration_eps = 25000

save_interval = 1000
model_name = "iddpg"

####
#### DDP ####
####
use_gpu = True # baselines
use_distributed = False
port = '11110'

cpu_processes = 0
gpu_process = 4
percentage_search = 0.05
####
#### Model Definition ####
####

# MAAC
critic_hidden_dim = 64
pol_hidden_dim = 64
attend_heads = 4
gamma = 0.99
tau = 0.001
q_lr = 0.001
pi_lr = 0.001
reward_scale = 100
batch_size = 1024

####
#### Replay Buffer ####
####
buffer_length = int(1e6)

####
#### Environment ####
####
env_id = "multi_speaker_listener" #simple_speaker_listener, simple_tag, simple_push, multi_speaker_listener, simple_spread, simple_push, simple_adversary, simple_tag, simple_crypto, fullobs_collect_treasure
n_episodes = 50000
n_rollout_threads = 32
n_training_threads = 1
n_updates = 4
steps_per_update = 100
episode_length = 25
discrete_action = True
init_noise_scale = 0.3
final_noise_scale = 0.0
seed = 1
n_exploration_eps = 50000

save_interval = 1000
model_name = "maac"

####
#### DDP ####
####
use_gpu = True # baselines
use_distributed = False
port = '11113'

cpu_processes = 0
gpu_process = 2
percentage_search = 0.05
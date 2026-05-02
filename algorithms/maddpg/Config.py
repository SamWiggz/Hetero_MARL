####
#### Model Definition ####
####

#MADDPG
hidden_sizes = [1024, 1024]
batch_size = 16384
gamma = 0.95
tau = 0.01
lr = 0.01
agent_alg = "MADDPG" # "DDPG for simple_tag"
adversary_alg = "MADDPG"

####
#### Replay Buffer ####
####
buffer_length = int(1e6)

####
#### Environment ####
####
env_id = "simple_tag"
# Only the dictionary entry matching env_id is used.
# Edit these counts here instead of editing files under multiagent/scenarios.
env_config = {
    "simple": {"num_agents": 1, "num_landmarks": 1},
    "simple_spread": {"num_agents": 8, "num_landmarks": 8},
    "simple_adversary": {"num_agents": 8, "num_adversaries": 1},
    "simple_tag": {"num_good_agents": 1, "num_adversaries": 3, "num_landmarks": 2},
    "simple_push": {"num_agents": 2, "num_adversaries": 1, "num_landmarks": 2},
    "simple_speaker_listener": {"num_agents": 2, "num_landmarks": 3},
    "simple_reference": {"num_agents": 2, "num_landmarks": 3},
    "simple_crypto": {"num_adversaries": 1, "num_good_listeners": 1, "num_speakers": 1, "num_landmarks": 2},
    "simple_world_comm": {"num_good_agents": 2, "num_adversaries": 4},
    "multi_speaker_listener": {"num_listeners": 8, "num_speakers": 8, "num_landmarks": 6},
    "fullobs_collect_treasure": {"num_agents": 4, "num_collectors": 3},
}
n_episodes = 25000
n_updates = 4
steps_per_update = 100
# Print average data-collection and model-update times every N model updates.
# Set to 0 to disable periodic timing summaries.
timing_log_interval_updates = 20
episode_length = 25
discrete_action = True
init_noise_scale = 0.3
final_noise_scale = 0.0
seed = 1
n_exploration_eps = 25000

####
#### System Parameters ####
####
port = '11110'
n_rollout_threads = 32
# (num_gpu_processes, num_cpu_processes, cores_per_cpu_process)
config = (2, 0, 0)

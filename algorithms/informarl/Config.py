####
#### Model Definition ####
####

# InforMARL
algorithm_name = "rmappo"
hidden_sizes = [64, 64]
lr = 7e-4
critic_lr = 7e-4
ppo_epoch = 10
gain = 0.01
use_valuenorm = True
use_ReLU = True

####
#### Replay Buffer / PPO Batch ####
####

batch_size = 128
num_mini_batch = 1
auto_mini_batch_size = True

####
#### Environment ####
####

env_name = "GraphMPE"
env_id = "graph_navigation" # formation, line, graph_navigation
project_name = "informarl"
experiment_name = "informarl"
user_name = "marl"
seed = 0
episode_length = 25
num_env_steps = 2000000
# Print average data-collection and model-update times every N model updates.
# Set to 0 to disable periodic timing summaries.
timing_log_interval_updates = 20
collision_rew = 5
use_cent_obs = False
graph_feat_type = "relative"
# InforMARL's imported train script still needs a Torch thread setting even
# when the Hetero_MARL layout has no CPU learner processes.
torch_threads = 1

# Only the dictionary entry matching env_id is used.
# Edit these counts/settings here instead of appending CLI args after python main.py.
env_config = {
    "formation": {
        "scenario_name": "simple_graph_formation",
        "num_agents": 3,
        "hidden_sizes": [64, 64],
    },
    "line": {
        "scenario_name": "line_graph",
        "num_agents": 7,
        "hidden_sizes": [128, 128],
    },
    "graph_navigation": {
        "scenario_name": "navigation_graph",
        "num_agents": 10,
        "hidden_sizes": [1024, 1024],
    },
}

####
#### System Parameters ####
####
port = "11110"
n_rollout_threads = 32
# (num_gpu_processes, num_cpu_processes, cores_per_cpu_process)
config = (2, 0, 0)

# Compatibility names used by the imported InforMARL training script.
gpu_process, cpu_processes, cores_per_cpu_process = config
n_training_threads = cores_per_cpu_process if cpu_processes > 0 else torch_threads


def get_env_settings(name=None):
    selected_env_id = name or env_id
    if selected_env_id not in env_config:
        valid = ", ".join(sorted(env_config))
        raise ValueError("Unknown InforMARL env_id '{}'. Valid env_id values: {}".format(selected_env_id, valid))

    selected_env = env_config[selected_env_id]
    selected_hidden_sizes = selected_env.get("hidden_sizes", hidden_sizes)
    return selected_env_id, {
        "project_name": project_name,
        "env_name": env_name,
        "algorithm_name": algorithm_name,
        "seed": seed,
        "experiment_name": experiment_name,
        "scenario_name": selected_env["scenario_name"],
        "num_agents": selected_env["num_agents"],
        "collision_rew": collision_rew,
        "n_training_threads": n_training_threads,
        "hidden_size": selected_hidden_sizes[0],
        "n_rollout_threads": n_rollout_threads,
        "num_mini_batch": num_mini_batch,
        "episode_length": episode_length,
        "num_env_steps": num_env_steps,
        "ppo_epoch": ppo_epoch,
        "gain": gain,
        "lr": lr,
        "critic_lr": critic_lr,
        "user_name": user_name,
        "use_cent_obs": use_cent_obs,
        "graph_feat_type": graph_feat_type,
        "target_mini_batch_size": batch_size,
    }


def build_argv(name=None):
    _, args = get_env_settings(name)
    argv = []

    # The upstream InforMARL parser uses store_false for these flags.
    if not use_valuenorm:
        argv.append("--use_valuenorm")
    if not use_ReLU:
        argv.append("--use_ReLU")
    if auto_mini_batch_size:
        argv.append("--auto_mini_batch_size")

    # InforMARL's original parser uses this flag to disable wandb.
    argv.append("--use_wandb")

    for key, value in args.items():
        argv.append("--{}".format(key))
        argv.append(str(value))

    return argv

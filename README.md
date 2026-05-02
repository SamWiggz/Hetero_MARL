# Hetero_MARL: Multi-Agent Reinforcement Learning on Heterogeneous Platforms

## Overview

## 1. Environment Setup
Preliminary: Have [conda](https://www.anaconda.com/download/success) installed <br />
Note: May need to download [torch](https://pytorch.org/get-started/locally/) manually

1. Clone this repository:
```
git clone https://github.com/SamWiggz/Hetero_MARL.git
```
2. Extract the provided environment file using conda:
```
conda env create --file=environment.yml
```
3. Activate conda environment:
```
conda activate hetero_marl
```
4. Install PyTorch
```
pip install torch==2.6.0+cu126 torchvision==0.21.0+cu126 torchaudio==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126
```

## 2. Running an Example
We provide four state-of-the-art MARL algorithms:
1. [Multi-Agent Deep Deterministic Policy Gradient](https://arxiv.org/pdf/1706.02275) (MADDPG)
2. [Multi-Actor-Attention-Critic](https://arxiv.org/pdf/1810.02912) (MAAC)
3. [Independent Deep Deterministic Policy Gradient](https://arxiv.org/pdf/1509.02971) (IDDPG)
4. [MARL through Intelligent Information Aggregation](https://arxiv.org/pdf/2211.02127) (InforMARL)

---
1. Go to the desired algorithm's directory. Example for MADDPG:
```
cd algorithms/maddpg
```
2. Edit the Config.py file to adjust desired hyperparameters and the Hetero_MARL execution layout.

Some Important hyperparameters:
- `n_episodes`: number of episodes for the experiment
- `n_rollout_threads`: max number of possible rollout simulations
- `hidden_sizes`: hidden dimension size for agent neural networks
- `batch_size`: batch size of experiences that will be sampled by learner(s)
- `env_config`: per-environment agent, role, and landmark counts used by `env_id`
- `config`: hardware configuration tuple: `(num_gpu_processes, num_cpu_processes, cores_per_cpu_process)`
3. Run your accelerated MARL experiment!
```
python main.py
```

## Acknowledgement
This work was supported by Intel Corporation and the National Science Foundation under grant OAC-2411446.

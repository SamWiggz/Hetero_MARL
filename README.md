# SMART: A Scalable Multi-Agent Reinforcement Learning Framework on Heterogeneous Platforms

* Note: This is a submission release that is not finalized

## Overview

## 1. Environment Setup
Preliminary: Have [conda](https://www.anaconda.com/download/success) installed
Note: May need to download [torch](https://pytorch.org/get-started/locally/) manually

1. Clone this repository:
```
git clone XXXXXXX(anonymous, manually download)
```
2. Extract the provided environment file using conda:
```
conda env create --file=environment.yml
```
3. Activate conda environment:
```
conda activate smart
```

## 2. Running an Example
We provide four state-of-the-art MARL algorithms:
1. [Multi-Agent Deep Deterministic Policy Gradient](https://arxiv.org/pdf/1706.02275) (MADDPG)
2. [Multi-Actor-Attention-Critic](https://arxiv.org/pdf/1810.02912) (MAAC)
3. [Independent Deep Deterministic Policy Gradient](https://arxiv.org/pdf/1509.02971) (IDDPG)
4. [MARL through Intelligent Information Aggregation](https://arxiv.org/pdf/2211.02127) (InforMARL)

---
1. Go to the desired algorithm's directory. Example for IDDPG:
```
cd algorithms/maddpg
```
2. Edit the Config.py file to adjust desired hyperparameters.

Some Important hyperparameters:
- `n_episodes`: number of episodes for the experiment
- `n_rollout_threads`: max number of possible rollout simulations
- `hidden_sizes`: hidden dimension size for agent neural networks
- `batch_size`: batch size of experiences that will be sampled by learner(s)
- `percentage_search`: percentage of the entire search space that SMART will test (default 0.05)
3. Run your SMART-enabled MARL experiment!
```
python main.py
```

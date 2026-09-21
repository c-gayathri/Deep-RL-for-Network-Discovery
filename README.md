# Deep Reinforcement Learning for Network Discovery

This repository contains a research implementation of deep reinforcement learning for sampling partially observed social networks. The learned policy selects nodes to query, expands the discovered subgraph, and evaluates the resulting sample through downstream influence maximization. The project also includes experiments that measure discovery and influence across communities.

The implementation builds on the method described in *Influence Maximization in Unknown Social Networks: Learning Policies for Effective Graph Sampling* ([AAMAS 2020 paper](https://arxiv.org/abs/1907.11625)). The original license is retained in [LICENSE](LICENSE).

## Repository layout

- `train.py` — latest influence-oriented training entry point.
- `train_fairness.py` — training entry point with community fairness metrics.
- `rl_alg/` — DQN trainer, replay buffer, and reinforcement-learning utilities.
- `expts/` — graph environment, influence estimation, and experiment helpers.
- `diffpool/` — graph neural-network and pooling components.
- `ge/` — graph embedding models and DeepWalk utilities.
- `data/` — input networks and graph parameter files used by the experiments.
- `scripts/` — synthetic graph generation and result plotting utilities.
- `results/figures/` — selected figures retained with the project.
- `docs/` — notes on the training pipeline and experiment outputs.

## Environment

The code was developed against the Python and machine-learning APIs available around 2021–2022. Python 3.8 is the safest starting point because newer NetworkX and Gensim releases removed APIs used here.

```bash
python3.8 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch installations can be platform- and CUDA-specific. If the installation from `requirements.txt` is unsuitable for the target machine, install the matching PyTorch build first and then install the remaining dependencies.

## Running an experiment

Run commands from the repository root so that package imports and data paths resolve correctly.

```bash
python train.py \
  --eps 100 \
  --use_cuda 0 \
  --write 0 \
  --newFile 1 \
  --fileName results/raw/example.txt
```

For the community-aware objective, use `train_fairness.py` with the same core arguments. Training checkpoints, logs, TensorBoard runs, and raw experiment output are intentionally ignored by Git.

To plot an output file:

```bash
python scripts/plot_results.py results/raw/example.txt
```

See [docs/TRAINING.md](docs/TRAINING.md) for the main parameters and the training flow.

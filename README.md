# Automata-Conditioned Cooperative Multi-Agent Reinforcement Learning (ACC-MARL)

This repo contains the code for our paper: [Automata-Conditioned Cooperative Multi-Agent Reinforcement Learning (ACC-MARL)](https://arxiv.org/pdf/2511.02304).

## Setup

Install helper packages: [DFAx](https://github.com/rad-dfa/dfax), [dfa-gym](https://github.com/rad-dfa/dfa-gym), and [rad-embeddings](https://github.com/rad-dfa/rad-embeddings). Then, proceed with this repo.

```
git clone https://github.com/rad-dfa/acc-marl.git
cd acc-marl
pip install -r requirements.txt
```

## Experiments

To reproduce the results in the paper, run [`exp_train.sh`](https://github.com/rad-dfa/acc-marl/blob/main/acc_marl/exp_train.sh) to train ACC-MARL policies, run [`exp_train_recurrent.sh`](https://github.com/rad-dfa/acc-marl/blob/main/acc_marl/exp_train_recurrent.sh) to train (baseline) recurrent policies, and run [`exp_test.sh`](https://github.com/rad-dfa/acc-marl/blob/main/acc_marl/exp_test.sh) to test trained policies. See [train_policy.py](https://github.com/rad-dfa/acc-marl/blob/main/acc_marl/train_policy.py), [train_recurrent\_policy.py](https://github.com/rad-dfa/acc-marl/blob/main/acc_marl/train_recurrent_policy.py), and [test_policy.py](https://github.com/rad-dfa/acc-marl/blob/main/acc_marl/test_policy.py) for more details.

## Citation

```
@article{DBLP:journals/corr/abs-2511-02304,
  author       = {Beyazit Yalcinkaya and
                  Marcell Vazquez{-}Chanlatte and
                  Ameesh Shah and
                  Hanna Krasowski and
                  Sanjit A. Seshia},
  title        = {Automata-Conditioned Cooperative Multi-Agent Reinforcement Learning},
  journal      = {CoRR},
  volume       = {abs/2511.02304},
  year         = {2025}
}
```




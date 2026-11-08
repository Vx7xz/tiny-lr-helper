# A tiny learned learning-rate helper

A helper of 22 learned numbers (`helper.pt`) picks the learning rate for every layer at every training step,
inside a single run, with no learning-rate search. Results against tuned AdamW are in
[X_report.md](X_report.md).

## How it works

`pg2_fast.py` does the per-step work:
1. For each tensor (a layer's weight or bias), the helper reads 13 inputs and computes
   `c = 1.5 * tanh(w . inputs + b)`.
2. The lr is `1e-3 * 10**c * cosine(t)`, times one whole-model factor `10**(z . delta)`. z is the
   parameter-weighted mean of the tensors' fixed inputs and the run length.
3. The update is Adam without weight decay, plus a weight average. Optimizer state is kept in bf16, about
   14 bytes per weight (AdamW: 16).

The 13 inputs are:
- training progress and run length;
- loss level and loss trend;
- gradient/momentum agreement and gradient-norm trend;
- the last lr;
- fan-in, layer position, and whether the tensor is a vector, a conv, an embedding or a norm.

The helper was meta-trained with evolution strategies for both accuracy and speed. The objective is the
validation loss across the run, with the end weighted most.

## Train / test split

`lr_tasks.py` keeps the two sets apart, and asserts that no dataset is in both.
- **Trained on:** MNIST, EMNIST, CIFAR-100, stdlib Python text, enwik8.
- **Tested on:** Fashion-MNIST, KMNIST, CIFAR-10, SVHN, TinyShakespeare, WikiText-2, Tiny ImageNet,
  TinyStories.

## Use it

```python
import torch
from pg_fast import train_pg_fast

w = torch.load("helper.pt")
train_pg_fast(model, batch_loss, T, w)  # batch_loss(t) returns the training loss of step t; T = total steps
```

Requirements: Python 3, PyTorch with CUDA, numpy, scipy, pillow, pyarrow, psutil, pynvml, torchvision.

## Reproduce the report

Put the datasets under `data/`, at the paths in `lr_tasks.py` and `final_test.py`.

```
python final_test.py --task cnn --data tinyimagenet --augment --average --seeds 3 --steps 3000 --pg helper.pt
python final_test.py --task transformer --data tinystories --average --seeds 3 --steps 3000 --pg helper.pt
python final_test.py --task cnn --data tinyimagenet --augment --seeds 3 --limits --pg helper.pt
python lr_choice_test2.py helper.pt --final
```

`logs/` holds the runs behind the report. In the logs the helper is called `pg2_v5.pt`. Rows of earlier
helper versions were removed from the AdamW baseline logs.

## Limits

This is early work, and the helper still needs more training:
- It was meta-trained on 5 small datasets and small models only.
- Its validation score was still moving between checkpoints when training stopped.
- On text, AdamW's best settings are still slightly ahead.
- On images, its loss is worse in normal use (overconfident).

Small models only: a CNN of about 0.3M parameters and a transformer of about 0.8M. Each result is 3 seeds.
It is not tested at large scale.

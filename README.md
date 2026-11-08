# Tiny LR Helper

A tiny learned helper of 22 numbers that picks the learning rate for every layer at every training step,
inside a single run, with no learning-rate search.

## Results vs tuned AdamW

Tested on two datasets the helper never saw during training: Tiny ImageNet (images, small CNN) and
TinyStories (text, small transformer). Every number is the average of 3 runs.

### Normal use
The helper is used as is. AdamW is tuned the usual way: 13 runs, one per learning rate, keeping the best.

| | Tuned AdamW | Helper |
|---|---|---|
| Tiny ImageNet accuracy | 66.9% | **71.6%** |
| Tiny ImageNet steps to reach the target | 1650 | **517** (3.2x faster) |
| TinyStories test loss (lower is better) | 0.980 | **0.972** |
| TinyStories character accuracy | 69.6% | **69.9%** |
| Total compute | 100% | **~7%** (1 run instead of 13 + 1) |
| GPU memory | 535 / 173 MB | 535 / 170 MB |

### Both pushed to their limit
- AdamW searched learning rate x weight decay x training length.
- The helper searched only one learning-rate multiplier x training length. It was not retrained.

The best setting was picked on validation data, then scored on test data.

| Best possible | AdamW | Helper |
|---|---|---|
| Tiny ImageNet accuracy | 70.1% | **73.6%** |
| Tiny ImageNet test loss | 2.54 | **2.28** |
| TinyStories test loss | **0.919** | 0.925 |
| TinyStories character accuracy | **71.4%** | 71.2% |
| Search runs (images / text) | 84 / 56 | **27 / 18** |

On images, the helper with no search at all (71.6%) already beats AdamW's best (70.1%). On text, AdamW's
best is slightly ahead.

Full report: [X_report.md](X_report.md). Raw runs: [logs/](logs).

## How it works

**The problem.** Training a network with AdamW needs a good learning rate, and finding it usually takes many
full training runs. The helper replaces that search.

**Each step, for each layer:**
1. The helper reads 13 simple signals about the layer and the training:
   - how far training is, and how long it is;
   - the loss level, and whether the loss is falling;
   - whether the gradient keeps its direction, and how the gradient size is changing;
   - the last learning rate it picked;
   - the layer's size, position and kind (weight or bias, conv, embedding, norm).
2. It multiplies each signal by a learned weight and adds them up: `c = 1.5 * tanh(w . signals + b)`.
3. The learning rate is `1e-3 * 10**c * cosine(t)`, times one learned factor for the whole model.
4. The weights are updated with Adam using that learning rate, and a running average of the weights is kept.
   Optimizer state is stored in bf16, about 14 bytes per weight (AdamW: 16).

So the whole "optimizer brain" is 22 numbers, and it costs almost nothing per step.

**How the 22 numbers were learned.** With evolution strategies:
1. Make small random changes to the numbers.
2. Train many small networks with each version.
3. Keep moving towards the versions whose networks ended with lower validation loss, and got there sooner.

It was trained only on MNIST, EMNIST, CIFAR-100, Python code text and enwik8. Testing used other datasets
only; `lr_tasks.py` enforces the split.

## Use it

```python
import torch
from pg_fast import train_pg_fast

w = torch.load("helper.pt")
train_pg_fast(model, batch_loss, T, w)  # batch_loss(t) returns the training loss of step t; T = total steps
```

Requirements: Python 3, PyTorch with CUDA, numpy, scipy, pillow, pyarrow, psutil, pynvml, torchvision.

To reproduce the report, put the datasets under `data/`, then:
```
python final_test.py --task cnn --data tinyimagenet --augment --average --seeds 3 --steps 3000 --pg helper.pt
python final_test.py --task transformer --data tinystories --average --seeds 3 --steps 3000 --pg helper.pt
python final_test.py --task cnn --data tinyimagenet --augment --seeds 3 --limits --pg helper.pt
```
In the logs the helper is called `pg2_v5.pt`.

## Limits

This is early work, and the helper still needs more training, on more and bigger datasets and models.
- Only small models were tested: a CNN of about 0.3M parameters and a transformer of about 0.8M.
- In normal use its image test loss is worse than AdamW's (overconfident), even though its accuracy is higher.
- On text, AdamW with weight averaging reaches the target a little sooner.

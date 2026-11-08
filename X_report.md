# A tiny learned helper that picks the learning rate, vs. tuned AdamW

**In short:** a helper with 22 learned numbers picks the learning rate by itself, within a single training run.
- On images it beats AdamW, even when both are pushed to their best settings: 73.6% vs 70.1%.
- On text it ties AdamW. AdamW's best is 0.6% better, and the helper's normal setting is 0.8% better.
- It uses about 7% of the compute of normal AdamW tuning.

**The idea.** Picking a good learning rate usually means trying many values. AdamW was tuned the usual way:
train 13 times, once per learning rate, and keep the best. The helper instead picks the learning rate for
every layer at every step, inside one training run.

**Fair test.** The helper was trained only on MNIST, EMNIST, CIFAR-100, Python code text and Wikipedia text
(enwik8). It was tested on two datasets it had never seen: Tiny ImageNet (images) and TinyStories (text).
Every number is the average of 3 runs.

## 1. Normal use (helper as is, AdamW tuned with 13 runs)

| | Tuned AdamW | Helper |
|---|---|---|
| Tiny ImageNet accuracy | 66.9% | **71.6%** |
| Tiny ImageNet steps to reach the target | 1650 | **517** (3.2x faster) |
| TinyStories test loss (lower is better) | 0.980 | **0.972** |
| TinyStories character accuracy | 69.6% | **69.9%** |
| Total compute | 100% | **~7%** (1 run instead of 13 + 1) |
| GPU memory | same | same |

## 2. Both pushed to their limit (the helper was not retrained)

Each method got a big search for its best settings:
- **AdamW:** learning rate x weight decay x training length, 84 runs on images and 56 on text.
- **Helper:** one learning-rate multiplier x training length, 27 and 18 runs.

The best setting was picked on validation data, then trained 3 times and scored on test data.

| Best possible | AdamW | Helper |
|---|---|---|
| Tiny ImageNet accuracy | 70.1% | **73.6%** |
| Tiny ImageNet test loss | 2.54 | **2.28** |
| TinyStories test loss | **0.919** | 0.925 |
| TinyStories character accuracy | **71.4%** | 71.2% |
| Search runs | 84 / 56 | **27 / 18** |

The helper's normal setting on images (71.6%, no search) is already above AdamW's best (70.1%).

## Honest notes
- This is early work: the helper still needs more training, on more and bigger datasets and models.
- These are small models (a CNN with ~0.3M parameters, a transformer with ~0.8M). It is not tested at
  large scale yet.
- In normal use the helper's image test loss is worse than AdamW's (it is overconfident), although its
  accuracy is higher. At the limit its loss is better too.
- On text, AdamW with weight averaging reaches the target a little sooner.
- Training the helper took about 10 GPU hours, once. Every training run after that saves the 13 tuning runs.

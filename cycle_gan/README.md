# cycle_gan

CycleGAN with Spectral Normalization and Class Activation Mapping Attention implemented using [PyTorch](https://pytorch.org/).

Based on the original MXNet implementation by [RangerUFO](https://github.com/ufownl/cycle_gan), converted to PyTorch for modern compatibility.

Trained on the [selfie2anime](https://www.kaggle.com/arnaud58/selfie2anime) dataset:

![p1](/docs/w1.jpg)
![p2](/docs/w2.jpg)
![p3](/docs/w3.jpg)
![p4](/docs/w4.jpg)
![p5](/docs/w5.jpg)
![p6](/docs/w6.jpg)
![p7](/docs/w7.jpg)
![p8](/docs/w8.jpg)
![p9](/docs/w9.jpg)

## Requirements

* [Python3](https://www.python.org/) (3.10+)
  * [PyTorch](https://pytorch.org/) (2.0+)
  * [torchvision](https://pytorch.org/)
  * [NumPy](https://www.numpy.org)
  * [opencv-python](https://github.com/skvark/opencv-python)
  * [Matplotlib](https://matplotlib.org/)
  * [Pillow](https://pillow.readthedocs.io/)

### Install

```
pip install -r requirements.txt
```

## Usage

### Weight files

Weight files use the `.pth` extension (PyTorch format):
- `model/<dataset>.gen_ab.pth` — Generator A→B
- `model/<dataset>.gen_ba.pth` — Generator B→A
- `model/<dataset>.dis_a.pth` — Discriminator A
- `model/<dataset>.dis_b.pth` — Discriminator B

### Train

```
python3 train.py --dataset selfie2anime --gpu
```

Details:

```
usage: train.py [-h] [--dataset DATASET] [--start_epoch START_EPOCH]
                [--max_epochs MAX_EPOCHS] [--lr_d LR_D] [--lr_g LR_G]
                [--batch_size BATCH_SIZE] [--lmda_cyc LMDA_CYC]
                [--lmda_idt LMDA_IDT] [--device_id DEVICE_ID] [--gpu]
```

### Test

```
python3 test.py --model selfie2anime /path/to/image.jpg
```

Details:

```
usage: test.py [-h] [--reversed] [--model MODEL] [--resize RESIZE]
               [--device_id DEVICE_ID] [--gpu]
               IMG [IMG ...]
```

### Run demo server

```
python3 server.py --model selfie2anime --port 8080
```

Details:

```
usage: server.py [-h] [--reversed] [--model MODEL] [--resize RESIZE]
                 [--addr ADDR] [--port PORT] [--device_id DEVICE_ID] [--gpu]
```

## GPU Support

The `--gpu` flag auto-detects the best available backend:
- NVIDIA GPU → CUDA
- Apple Silicon → MPS
- Otherwise → CPU fallback

## References

* [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://junyanz.github.io/CycleGAN/)
* [Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/abs/1802.05957)
* [U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation](https://arxiv.org/abs/1907.10830)

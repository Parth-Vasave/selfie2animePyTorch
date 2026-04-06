# selfie2animePyTorch

Convert selfie photos to anime-style images using a CycleGAN model implemented in PyTorch.

## Quick Start (Google Colab)

Open `selfie2anime_colab.ipynb` in Google Colab to try it instantly — no local setup needed.

## Local Usage

### Setup
```bash
cd cycle_gan
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run inference
```bash
python test.py --model selfie2anime your_selfie.jpg
```

Add `--gpu` for GPU acceleration (CUDA or Apple Silicon MPS).

### Train from scratch
```bash
python train.py --dataset selfie2anime --gpu
```

## Project Structure

```
cycle_gan/
  pix2pix_gan.py     # ResnetGenerator + PatchDiscriminator with CAM & spectral norm
  train.py           # Training loop
  test.py            # Inference & visualization
  server.py          # HTTP demo server
  dataset.py         # Dataset loading & augmentation
  image_pool.py      # Image buffer for training stability
  convert_weights.py # MXNet .params -> PyTorch .pth converter
  model/             # Pre-trained weights
```

## Architecture

- **Generator**: ResNet-based with 9 residual blocks, Class Activation Mapping (CAM), spectral normalization
- **Discriminator**: PatchGAN with CAM
- **Loss**: Adversarial (BCE) + Cycle consistency (L1) + Identity (L1)

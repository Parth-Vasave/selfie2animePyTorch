# selfie2animePyTorch

Convert selfie photos to anime-style images using deep learning with PyTorch.

## Quick Start (Google Colab)

Open `selfie2anime_colab.ipynb` in Google Colab to try it instantly — no local setup needed.

## Web Dashboard

Run the web dashboard locally for a drag-and-drop browser experience:

```bash
cd cycle_gan
pip install -r requirements.txt
python app.py --port 8080 --gpu
```

Then open http://localhost:8080 in your browser.

## CLI Usage

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
  app.py               # Flask web dashboard
  templates/index.html  # Dashboard frontend
  style_generator.py    # Generator architecture
  pix2pix_gan.py        # CycleGAN components (training)
  train.py              # Training loop
  test.py               # CLI inference & visualization
  dataset.py            # Dataset loading & augmentation
  image_pool.py         # Image buffer for training stability
  model/                # Pre-trained weights
```

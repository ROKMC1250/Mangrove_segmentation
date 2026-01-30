# MANGO: A Global Single-Date Paired Dataset for Mangrove Segmentation

<!-- **IEEE International Geoscience and Remote Sensing Symposium (IGARSS), 2026 (Under Review)** -->

This repository contains the official code for **MANGO**, a global single-date paired dataset for mangrove segmentation. This repository not only provides the MANGO dataset but also offers a comprehensive benchmark environment for training and evaluating various segmentation models on mangrove detection tasks.

[![arXiv](https://img.shields.io/badge/arXiv-PDF-b31b1b.svg)](https://arxiv.org/pdf/2601.17039) [![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow.svg)](https://huggingface.co/datasets/hjh1037/MANGO) [![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

[Junhyuk Heo](https://rokmc1250.github.io) | Beomkyu Choi | Hyunjin Shin | Darongsae Kwon

## Supported Models

This benchmark supports the following segmentation architectures:

| Model | Description |
|-------|-------------|
| UNet++ | Nested U-Net with dense skip connections |
| MAnet | Multi-scale Attention Network |
| PAN | Pyramid Attention Network |
| DeepLabV3+ | Encoder-Decoder with Atrous Separable Convolution |
| Segformer | Simple and Efficient Design for Semantic Segmentation with Transformers |
| FPN | Feature Pyramid Network |
| DPT | Dense Prediction Transformer |
| UperNet | Unified Perceptual Parsing Network |

## Setup

### Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies include:
- PyTorch >= 2.0.0
- segmentation-models-pytorch
- rasterio
- tensorboard

## Download Datasets

The MANGO dataset is available on Hugging Face:

**[https://huggingface.co/datasets/hjh1037/MANGO](https://huggingface.co/datasets/hjh1037/MANGO)**

### Dataset Statistics

| Split | Images | Masks |
|-------|--------|-------|
| Train | 34,272 | 34,272 |
| Validation | 4,159 | 4,159 |
| Test | 4,272 | 4,272 |
| **Total** | **42,703** | **42,703** |

- **Image size**: 256 x 256 pixels
- **Image format**: GeoTIFF (13 bands, uint16)
- **Mask format**: GeoTIFF (1 band, uint8, binary)

### Sentinel-2 Band Information

The dataset contains 13-band Sentinel-2 imagery with the following bands:

| Band Index | Band Name | Description | Resolution | Wavelength |
|------------|-----------|-------------|------------|------------|
| 1 | B1 | Aerosol | 60m | 443nm |
| 2 | B2 | Blue | 10m | 490nm |
| 3 | B3 | Green | 10m | 560nm |
| 4 | B4 | Red | 10m | 665nm |
| 5 | B5 | Red Edge 1 | 20m | 705nm |
| 6 | B6 | Red Edge 2 | 20m | 740nm |
| 7 | B7 | Red Edge 3 | 20m | 783nm |
| 8 | B8 | NIR | 10m | 842nm |
| 9 | B8A | Narrow NIR | 20m | 865nm |
| 10 | B9 | Water Vapor | 60m | 945nm |
| 11 | B11 | SWIR 1 | 20m | 1610nm |
| 12 | B12 | SWIR 2 | 20m | 2190nm |
| 13 | SCL | Scene Classification | 20m | - |

> **RGB Visualization**: To visualize images as true-color RGB, use bands **4 (Red)**, **3 (Green)**, **2 (Blue)**.

### Directory Structure

After downloading, place the dataset in the `datasets/GEE/` directory:

```
datasets/
└── GEE/
    └── MANGO/
        ├── train/
        │   ├── images/
        │   └── masks/
        ├── validation/
        │   ├── images/
        │   └── masks/
        └── test/
            ├── images/
            └── masks/
```

If you place the dataset in a different location, update the `data/root_dir` field in the experiment config file:

```yaml
data:
  root_dir: 'your/custom/path'
```

## Training

To train a model, use the `train.py` script with a configuration file:

```bash
python train.py --config config/experiment_config/unetpp_MVI.yaml
```

### Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Path to YAML configuration file | `config/mangrove_config.yaml` |
| `--uid` | Unique identifier for the experiment | Auto-generated |
| `--model-path` | Path to checkpoint for resuming training | None |
| `--patience` | Patience for early stopping (-1 to disable) | -1 |
| `--seed` | Random seed for reproducibility | Random |
| `--port` | Port for DDP (Distributed Data Parallel) | Random |
| `--no-ddp` | Disable DDP and use single GPU | False |
| `--no-save` | Disable logging and checkpoint saving (for debugging) | False |

### Example Commands

```bash
# Train UNet++ with default settings
python train.py --config config/experiment_config/unetpp_MF.yaml

# Train with custom experiment ID and seed
python train.py --config config/experiment_config/segformer_MF.yaml --uid my_experiment --seed 42

# Train on single GPU without DDP
python train.py --config config/experiment_config/dpt_MF.yaml --no-ddp

# Resume training from checkpoint
python train.py --config config/experiment_config/unetpp_MF.yaml --model-path logs/unetpp_MF_v1/weights/last.pt
```

Training logs and checkpoints are saved to `logs/{uid}/`:
```
logs/
└── {uid}/
    ├── config.yaml
    ├── weights/
    │   ├── best.pt
    │   └── last.pt
    └── tensorboard/
```

## Test

To evaluate a trained model on the test dataset, use the `test.py` script:

```bash
python test.py logs/{uid}
```

### Test Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `log_dir` | Path to the log directory containing trained model (required) | - |
| `--weight-type` | Type of weights to load: `best` or `last` | `last` |
| `--batch-size` | Batch size for testing | 12 |
| `--num-workers` | Number of workers for data loading | 4 |

### Example Commands

```bash
# Test with best checkpoint
python test.py logs/unetpp_MF_v1 --weight-type best

# Test with custom batch size
python test.py logs/segformer_MF_v1 --weight-type last --batch-size 8
```

Test results are saved to `logs/{uid}/results/`:
```
results/
├── detailed_results.csv    # Per-image metrics
├── summary_results.csv     # Aggregated statistics
├── visualization/          # Visual comparisons
└── predictions/            # Binary prediction masks
```

## Predict

The `predict.py` script is designed for inference on real-world satellite images that exceed the 256x256 training patch size and do not have ground truth labels.

Given a 13-channel Sentinel-2 GeoTIFF image, it produces a binary mangrove segmentation mask using sliding window inference with Gaussian blending for smooth transitions.

### Predict Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--image_path` | Path to input 13-channel TIF image | Required |
| `--logs_dir` | Path to logs directory containing trained models | `logs` |
| `--output_path` | Path to save the comparison figure | Auto-generated |
| `--patch_size` | Patch size for sliding window | 256 |
| `--overlap` | Overlap ratio between patches (0-1) | 0.5 |
| `--tta` | Enable Test Time Augmentation | False |
| `--gpus` | GPU IDs to use | `0` |

### Example Commands

```bash
# Basic prediction
python predict.py --image_path path/to/sentinel2_image.tif --logs_dir logs

# Prediction with TTA and higher overlap for smoother results
python predict.py --image_path path/to/image.tif --logs_dir logs --overlap 0.75 --tta

# Specify output path
python predict.py --image_path path/to/image.tif --output_path results/my_prediction.png
```

The script automatically:
1. Loads all trained models from the `logs_dir`
2. Runs inference with both `best` and `last` checkpoints
3. Generates a comparison figure showing predictions from all models
4. Saves individual prediction masks to `results/individual_predictions/`

<!-- ## Citation

If you use this dataset or code in your research, please cite: -->

<!-- ```bibtex
@inproceedings{heo2026mango,
  title={MANGO: A Global Single-Date Paired Dataset for Mangrove Segmentation},
  author={Heo, Junhyuk and Choi, Beomkyu and Shin, Hyunjin and Kwon, Darongsae},
  booktitle={IEEE International Geoscience and Remote Sensing Symposium (IGARSS)},
  year={2026}
}
``` -->

## Acknowledgements

This code is mainly based on [Quick-Torch](https://github.com/SteveImmanuel/quick-torch).

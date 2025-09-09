# Image-to-Text with Energy-Based Models

This project implements and trains an **energy-based model (EBM)** for structured sequence decoding tasks using synthetic word images. It demonstrates how EBMs can be used to infer discrete structured outputs—like character sequences—from high-dimensional inputs such as images, using differentiable energy functions and dynamic programming. Instead of predicting characters with a softmax at each time step, we learn an energy landscape over latent “window × class” decisions, and then use dynamic programming (DP) to find low-energy alignments and to decode full words.

---

## 🔍 Project Overview

- **Model**: A simple CNN-based encoder trained to minimize energy along the correct label path, using a custom loss.
- **Data**: Synthetic images of lowercase words generated via `SimpleWordsDataset`.
- **Training**: Supervised learning with cross-entropy loss over optimal alignment paths.
- **Inference**: Recovers the most likely character sequence by decoding energy maps using dynamic programming.
- **Visualization**: Path matrix and energy heatmaps are saved as images.

This project was originally inspired by a homework assignment but expanded into a general-purpose EBM pipeline for experimenting with structured decoding.

---

## Quick Start for MacOS

### 1. Clone the repository
git clone https://github.com/your-username/energy-based-model-for-structured-prediction.git

cd energy-based-model-for-structured-prediction

### 2. Set up your environment
python -m venv venv

source venv/bin/activate

pip install -r requirements.txt

### 3. Run training (not necessary)
python main.py --epochs 1 --batch-size 32

### 4. Run inference
python main.py --skip-train --checkpoint outputs/checkpoints/best.pth --word "your-words"

---
```bash
Repository Structure
├── main.py           # Entry point: trains (N epochs), logs char acc, then decodes --word
├── train.py          # Trainer + DP utilities + decoder + logger + config snapshot
│   ├── train_ebm_model()     # One-epoch training loop (backprop on DP-aligned CE)
│   ├── build_ce_matrix()     # [B,L,T] CE from energies and target indices
│   ├── find_path()           # DP to find minimal-cost path over CE (free energy)
│   ├── path_cross_entropy()  # Sum along the DP path (the training loss)
│   ├── decode_segmented()    # DP segmentation + per-segment letter choice (inference)
│   ├── evaluate_char_acc()   # Char accuracy on a loader using decode_segmented
│   ├── TrainLogger           # CSV logger + simple PNG plot (no pandas dependency)
│   └── save_config()         # Dump run config to outputs/config.json
├── model.py          # SimpleNet CNN
├── data.py           # SimpleWordsDataset: renders synthetic word images
├── utils.py          # transform_word() + collate_fn() (returns images, targets, raw_texts)
├── inference.py      # plot utilities: energy heatmap + path (for gold targets)
├── config.py         # token constants (ALPHABET_SIZE, BETWEEN, DASH) + FONT_PATH
├── outputs/          # logs, plots, checkpoints
└── requirements.txt  # dependencies
```

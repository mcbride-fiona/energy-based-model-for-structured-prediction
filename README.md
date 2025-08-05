# Image-to-Text with Energy-Based Models

This project implements and trains an **energy-based model (EBM)** for structured sequence decoding tasks using synthetic word images. It demonstrates how EBMs can be used to infer discrete structured outputs—like character sequences—from high-dimensional inputs such as images, using differentiable energy functions and dynamic programming.

---

## 🔍 Project Overview

- **Model**: A simple CNN-based encoder trained to minimize energy along the correct label path, using a custom loss.
- **Data**: Synthetic images of lowercase words generated via `SimpleWordsDataset`.
- **Training**: Supervised learning with cross-entropy loss over optimal alignment paths.
- **Inference**: Recovers the most likely character sequence by decoding energy maps using dynamic programming.
- **Visualization**: Path matrix and energy heatmaps are saved as images.

This project was originally inspired by a homework assignment but expanded into a general-purpose EBM pipeline for experimenting with structured decoding.

---

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/your-username/energy-based-model-for-structured-prediction.git
cd energy-based-model-for-structured-prediction

### 2. Set up your environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

### 3. Run training and inference
python main.py --word hello 

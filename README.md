# AAINA Skin Type Classifier

On-device skin type classification model for the AAINA skincare app. Classifies facial skin into three categories: **Acne**, **Dry**, and **Oily** using handcrafted visual features and an SVM classifier.

## Overview

This model powers AAINA's face scan pipeline by detecting the user's skin type from a facial image. Instead of relying on deep learning (which needs large datasets and GPU training), it uses a feature engineering approach with classical ML, making it lightweight, interpretable, and effective on small datasets.

**Results:**
| Metric | Value |
|--------|-------|
| 5-Fold CV Accuracy | 83.59% (+/- 3.02%) |
| Test Accuracy | 82.14% |
| Test Precision | 84.82% |
| Test F1 Score | 0.82 |

## How It Works

The pipeline extracts 37 handcrafted features from each facial image, then feeds them into an SVM with an RBF kernel.

**Features extracted (37 total):**
- Color channel means and standard deviations (6)
- RGB color histograms, 8 bins per channel (24)
- Gradient magnitude stats for texture detection (2)
- Local variance stats for skin roughness (2)
- Color ratios: R/G, R/B, G/B (3)

**Why these features work:**
- Oily skin reflects more light, showing up as higher brightness variance and shifted color histograms
- Dry skin has more texture (higher local variance and gradient magnitude)
- Acne-prone skin shows localized color irregularities and distinct red channel patterns

## Project Structure

```
.
├── README.md
├── requirements.txt
├── aaina_skin_classifier_train.py    # Training script
├── aaina_skin_model.pkl              # Trained model (SVM + scaler + encoder)
├── aaina_results.json                # Metrics and evaluation results
└── skin_dataset/                     # Dataset (not included, see below)
    ├── train/
    │   ├── acne/    (237 images)
    │   ├── dry/     (226 images)
    │   └── oil/     (226 images)
    ├── valid/
    │   ├── acne/    (20 images)
    │   ├── dry/     (19 images)
    │   └── oil/     (19 images)
    └── test/
        ├── acne/    (10 images)
        ├── dry/     (9 images)
        └── oil/     (9 images)
```

## Setup

**Requirements:** Python 3.10+

```bash
pip install -r requirements.txt
```

## Training

1. Download the [Roboflow SkinClassification Dataset](https://universe.roboflow.com/skincareexperiments/skinclassification-kyxvj/dataset/1) (775 images, 3 classes)
2. Place it in a `skin_dataset/` folder with `train/`, `valid/`, `test/` splits
3. Run:

```bash
python aaina_skin_classifier_train.py
```

Output will be saved to `model_output/`:
- `model.pkl` -- trained SVM model with scaler and label encoder
- `results.json` -- full evaluation metrics

## Inference

```python
import pickle
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter

IMG_SIZE = 64

def extract_features(img_path):
    img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    features = []

    for c in range(3):
        features.append(arr[:,:,c].mean())
        features.append(arr[:,:,c].std())

    for c in range(3):
        hist, _ = np.histogram(arr[:,:,c], bins=8, range=(0,1))
        features.extend(hist / hist.sum())

    gray = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]
    gy, gx = np.diff(gray, axis=0), np.diff(gray, axis=1)
    grad = np.sqrt(gy[:,:-1]**2 + gx[:-1,:]**2)
    features.extend([grad.mean(), grad.std()])

    local_mean = uniform_filter(gray, size=7)
    local_var = uniform_filter(gray**2, size=7) - local_mean**2
    features.extend([local_var.mean(), local_var.std()])

    rm, gm, bm = arr[:,:,0].mean(), arr[:,:,1].mean(), arr[:,:,2].mean()
    features.extend([rm/(gm+1e-7), rm/(bm+1e-7), gm/(bm+1e-7)])

    return np.array(features, dtype=np.float32)

# Load model
with open("aaina_skin_model.pkl", "rb") as f:
    bundle = pickle.load(f)

# Predict
features = bundle["scaler"].transform([extract_features("path/to/face.jpg")])
skin_type = bundle["label_encoder"].inverse_transform(bundle["model"].predict(features))[0]
print(f"Skin Type: {skin_type}")  # "acne", "dry", or "oil"
```

## Model Details

| Parameter | Value |
|-----------|-------|
| Algorithm | SVM (RBF kernel) |
| C | 1.0 |
| Gamma | scale |
| Feature count | 37 |
| Image input size | 64x64 RGB |
| Training samples | 689 |
| Classes | acne, dry, oil |

**Per-class performance (test set):**

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Acne | 0.80 | 0.80 | 0.80 |
| Dry | 1.00 | 0.67 | 0.80 |
| Oily | 0.75 | 1.00 | 0.86 |

## Dataset

[Roboflow SkinClassification Image Dataset](https://universe.roboflow.com/skincareexperiments/skinclassification-kyxvj/dataset/1) - 775 facial images across 3 skin type categories. Images were auto-oriented and resized during preprocessing. The dataset is not included in this repository.

## Future Work

- Expand to multi-label skin concern detection (acne, dark spots, dark circles, redness, pigmentation)
- Integrate zone-based classification (T-Zone, U-Zone, C-Zone) for combination skin detection
- Convert to CoreML for native on-device inference in the iOS app
- Train with larger datasets (GlowMix, Nexdata) for improved generalization

## License

MIT

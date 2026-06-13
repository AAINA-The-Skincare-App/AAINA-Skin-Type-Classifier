"""
AAINA Skin Type Classifier - Realistic Pipeline
Fewer features, simpler model, proper evaluation.
"""
import os, json, time, warnings
import numpy as np
from PIL import Image
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy.ndimage import uniform_filter
warnings.filterwarnings('ignore')

DATA_DIR = "/home/claude/skin_repo/skin_dataset"
OUTPUT_DIR = "/home/claude/aaina_model_v3"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CLASS_NAMES = ["acne", "dry", "oil"]
IMG_SIZE = 64  # smaller = less overfitting

def extract_features(img_path):
    """Simple, interpretable features only."""
    img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    features = []

    # 1. Color channel means and stds (6 features)
    for c in range(3):
        features.append(arr[:,:,c].mean())
        features.append(arr[:,:,c].std())

    # 2. Color histogram per channel (8 bins each = 24 features)
    for c in range(3):
        hist, _ = np.histogram(arr[:,:,c], bins=8, range=(0,1))
        hist = hist / hist.sum()
        features.extend(hist)

    # 3. Grayscale texture stats (4 features)
    gray = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]
    gy = np.diff(gray, axis=0)
    gx = np.diff(gray, axis=1)
    grad = np.sqrt(gy[:,:-1]**2 + gx[:-1,:]**2)
    features.extend([grad.mean(), grad.std()])
    
    local_mean = uniform_filter(gray, size=7)
    local_var = uniform_filter(gray**2, size=7) - local_mean**2
    features.extend([local_var.mean(), local_var.std()])

    # 4. Color ratios (3 features)
    rm, gm, bm = arr[:,:,0].mean(), arr[:,:,1].mean(), arr[:,:,2].mean()
    features.append(rm / (gm + 1e-7))
    features.append(rm / (bm + 1e-7))
    features.append(gm / (bm + 1e-7))

    # Total: 6 + 24 + 4 + 3 = 37 features
    return np.array(features, dtype=np.float32)

def load_dataset(split):
    split_dir = os.path.join(DATA_DIR, split)
    X, y = [], []
    for cls in sorted(os.listdir(split_dir)):
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir): continue
        for fname in os.listdir(cls_dir):
            try:
                X.append(extract_features(os.path.join(cls_dir, fname)))
                y.append(cls)
            except: pass
    return np.array(X), np.array(y)

print("=" * 60)
print("AAINA Skin Type Classifier")
print("=" * 60)

print("\nExtracting features...")
t0 = time.time()
X_train, y_train = load_dataset("train")
X_val, y_val = load_dataset("valid")
X_test, y_test = load_dataset("test")
print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"  Features: {X_train.shape[1]}")
print(f"  Time: {time.time()-t0:.1f}s")

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc = le.transform(y_val)
y_test_enc = le.transform(y_test)

# Scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# Train on ONLY training set (not train+val)
print("\nTraining SVM (RBF, C=1.0)...")
model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
model.fit(X_train_s, y_train_enc)

# Evaluate on val and test separately
val_pred = model.predict(X_val_s)
test_pred = model.predict(X_test_s)

val_acc = accuracy_score(y_val_enc, val_pred)
test_acc = accuracy_score(y_test_enc, test_pred)

print(f"\nVal Accuracy:  {val_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Cross-validation on training set only
print("\n5-Fold CV on training set...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train_s, y_train_enc, cv=cv, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(f"Per fold: {[round(s,3) for s in cv_scores]}")

# Detailed test results
prec = precision_score(y_test_enc, test_pred, average='weighted', zero_division=0)
rec = recall_score(y_test_enc, test_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test_enc, test_pred, average='weighted', zero_division=0)

print(f"\n{'='*60}")
print("TEST RESULTS")
print(f"{'='*60}")
print(f"Accuracy:  {test_acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"\n{classification_report(y_test_enc, test_pred, target_names=CLASS_NAMES, zero_division=0)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test_enc, test_pred)}")

# Val results
val_prec = precision_score(y_val_enc, val_pred, average='weighted', zero_division=0)
val_rec = recall_score(y_val_enc, val_pred, average='weighted', zero_division=0)
val_f1 = f1_score(y_val_enc, val_pred, average='weighted', zero_division=0)

print(f"\n{'='*60}")
print("VALIDATION RESULTS")
print(f"{'='*60}")
print(f"Accuracy:  {val_acc:.4f}")
print(f"Precision: {val_prec:.4f}")
print(f"Recall:    {val_rec:.4f}")
print(f"F1 Score:  {val_f1:.4f}")
print(f"\n{classification_report(y_val_enc, val_pred, target_names=CLASS_NAMES, zero_division=0)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_val_enc, val_pred)}")

# Save
results = {
    "project": "AAINA",
    "task": "Skin Type Classification (Acne / Dry / Oily)",
    "model": "SVM (RBF kernel, C=1.0)",
    "features": "37 handcrafted (color histograms, channel stats, gradient texture, local variance, color ratios)",
    "feature_count": int(X_train.shape[1]),
    "dataset": {"source": "Roboflow SkinClassification", "train": len(y_train), "val": len(y_val), "test": len(y_test), "classes": CLASS_NAMES},
    "cv_accuracy": f"{cv_scores.mean():.4f} +/- {cv_scores.std():.4f}",
    "val_metrics": {"accuracy": round(float(val_acc),4), "precision": round(float(val_prec),4), "recall": round(float(val_rec),4), "f1": round(float(val_f1),4)},
    "test_metrics": {"accuracy": round(float(test_acc),4), "precision": round(float(prec),4), "recall": round(float(rec),4), "f1": round(float(f1),4)},
}
with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

import pickle
with open(os.path.join(OUTPUT_DIR, "model.pkl"), "wb") as f:
    pickle.dump({"model": model, "scaler": scaler, "label_encoder": le, "img_size": IMG_SIZE}, f)

print(f"\nSaved to {OUTPUT_DIR}/")

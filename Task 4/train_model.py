"""
train_model.py  (debug‑friendly, LinearSVC version)
---------------------------------------------------
add:  --dry-run      just counts images and exits
add:  tqdm           for live progress
"""

import os, re, cv2, joblib, argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from tqdm import tqdm

# ----------------------------------------------------------------------
# 1. Configurable paths and settings
# ----------------------------------------------------------------------
DATASET_DIR = r"dataset"
MODEL_PATH  = r"model/gesture_model.pkl"
IMG_SIZE    = 64
TEST_SPLIT  = 0.20
SEED        = 42

# ----------------------------------------------------------------------
# 2. CLI helpers
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true", help="just count images then quit")
args = parser.parse_args()

# ----------------------------------------------------------------------
# 3. Load dataset
# ----------------------------------------------------------------------
data, labels = [], []
class_pattern = re.compile(r"^(\d{1,2})")  # Match class directories like 00_palm

print(f"[INFO] Scanning '{DATASET_DIR}' …")
for root, dirs, files in os.walk(DATASET_DIR):
    dirname = os.path.basename(root)
    m = class_pattern.match(dirname)
    if not m:
        continue  # Skip folders that aren't gesture class dirs
    label = int(m.group(1))

    for fname in tqdm(files, desc=f"class {label:02}", leave=False):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        img_path = os.path.join(root, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append(img.flatten())
        labels.append(label)

n_images = len(data)
print(f"[INFO] Total images found: {n_images}")

if args.dry_run:
    exit(0)

if n_images == 0:
    raise RuntimeError(f"No images found in '{DATASET_DIR}'. Check your dataset path.")

# ----------------------------------------------------------------------
# 4. Train classifier using LinearSVC (faster than SVC)
# ----------------------------------------------------------------------
X = np.array(data, dtype=np.float32) / 255.0
y = np.array(labels, dtype=np.int32)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=TEST_SPLIT, stratify=y, random_state=SEED
)

print("[INFO] Training LinearSVC …")
clf = LinearSVC(max_iter=10000, dual=False)  # dual=False is better for n_samples > n_features
clf.fit(X_tr, y_tr)

print(f"[RESULT] Train acc: {clf.score(X_tr, y_tr):.3f}")
print(f"[RESULT] Val.  acc: {clf.score(X_te, y_te):.3f}")

# ----------------------------------------------------------------------
# 5. Save model
# ----------------------------------------------------------------------
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(clf, MODEL_PATH)
print(f"[OK] Model saved to {MODEL_PATH}")

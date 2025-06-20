import os, glob, joblib, random
import numpy as np
from tqdm import tqdm
from PIL import Image
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# ------- CONFIG --------------------------------------------------------------
DATA_DIR   = "train"          # Kaggle folder with dog.1234.jpg / cat.1234.jpg
MODEL_PATH = "model.joblib"   # output
IMAGE_SIZE = (128, 128)       # resize to this before HOG
PER_CLASS  = 6000             # sample this many dogs + cats to keep RAM low
# -----------------------------------------------------------------------------


def load_img(path):
    img = Image.open(path).convert("L").resize(IMAGE_SIZE)  # grayscale
    return np.array(img)


def extract_hog(img):
    return hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )


def build_dataset():
    cat_paths = random.sample(
        glob.glob(os.path.join(DATA_DIR, "cat.*.jpg")), PER_CLASS
    )
    dog_paths = random.sample(
        glob.glob(os.path.join(DATA_DIR, "dog.*.jpg")), PER_CLASS
    )
    X, y = [], []

    for label, paths in enumerate([cat_paths, dog_paths]):  # 0 = cat, 1 = dog
        for p in tqdm(paths, desc=f"Extracting {'cats' if label==0 else 'dogs'}"):
            img = load_img(p)
            feat = extract_hog(img)
            X.append(feat)
            y.append(label)

    return np.array(X), np.array(y)


def main():
    print("Building dataset ...")
    X, y = build_dataset()

    # train / val split
    split = int(0.8 * len(X))
    perm = np.random.permutation(len(X))
    X_train, y_train = X[perm[:split]], y[perm[:split]]
    X_val,   y_val   = X[perm[split:]], y[perm[split:]]

    print("Training Linear SVM ...")
    clf = LinearSVC(C=1.0, max_iter=10_000)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_val)
    print(f"Validation accuracy: {accuracy_score(y_val, preds):.3f}")

    joblib.dump(
        {
            "model": clf,
            "image_size": IMAGE_SIZE,
            "hog_params": {
                "orientations": 9,
                "pixels_per_cell": (8, 8),
                "cells_per_block": (2, 2),
                "block_norm": "L2-Hys",
            },
        },
        MODEL_PATH,
    )
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()

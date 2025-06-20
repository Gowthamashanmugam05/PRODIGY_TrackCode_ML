from flask import Flask, render_template, request, redirect, url_for
import numpy as np, joblib, tempfile, os
from PIL import Image
from skimage.feature import hog

app = Flask(__name__)
MODEL_INFO = joblib.load("model.joblib")
CLF        = MODEL_INFO["model"]
SIZE       = MODEL_INFO["image_size"]
HOG_KWARGS = MODEL_INFO["hog_params"]


def preprocess(file_storage):
    img = Image.open(file_storage.stream).convert("L").resize(SIZE)
    arr = np.array(img)
    feat = hog(arr, **HOG_KWARGS, feature_vector=True)
    return feat.reshape(1, -1)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files or request.files["image"].filename == "":
            return redirect(url_for("index"))

        feat = preprocess(request.files["image"])
        pred = CLF.predict(feat)[0]
        label = "Dog" if pred == 1 else "Cat"
        return render_template("result.html", label=label)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)

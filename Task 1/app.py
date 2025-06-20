from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("model.joblib")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        sqft = float(request.form["sqft"])
        bedrooms = float(request.form["bedrooms"])
        bathrooms = float(request.form["bathrooms"])

        # ❗ Fix: Pass DataFrame with expected column names
        X = pd.DataFrame([[sqft, bedrooms, bathrooms]],
                         columns=["GrLivArea", "BedroomAbvGr", "FullBath"])

        price = model.predict(X)[0]
        prediction = f"${price:,.0f}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)

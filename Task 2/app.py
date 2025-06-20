from flask import Flask, render_template, abort
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")  # Use a non‑GUI backend (important on servers)
import matplotlib.pyplot as plt
import io, base64, os

app = Flask(__name__)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Mall_Customers.csv")

if not os.path.isfile(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

# -----------------------------------------------------------------------------
# Helper function
# -----------------------------------------------------------------------------

def load_and_cluster(k: int = 5):
    """Load dataset, run K‑Means, and return a base64‑encoded plot + cluster centres."""

    # ---------------------------------------------------------------------
    # Read data & pick features
    # ---------------------------------------------------------------------
    df = pd.read_csv(DATA_PATH)
    X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

    # ---------------------------------------------------------------------
    # Standardise features and fit K‑Means (n_init=10 for broad compatibility)
    # ---------------------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    df["Cluster"] = labels

    # ---------------------------------------------------------------------
    # Build scatter plot
    # ---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(
        df["Annual Income (k$)"],
        df["Spending Score (1-100)"],
        c=labels,
        s=50,
        alpha=0.8,
    )
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1‑100)")
    ax.set_title(f"Customer Segments (k = {k})")
    fig.tight_layout()

    # ---------------------------------------------------------------------
    # Encode PNG to base64 so it can be embedded directly in the template
    # ---------------------------------------------------------------------
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)

    # ---------------------------------------------------------------------
    # Reverse the standardisation so centres are in original units
    # ---------------------------------------------------------------------
    centres_original = scaler.inverse_transform(kmeans.cluster_centers_)
    return img_b64, centres_original

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.route("/")
@app.route("/clusters/<int:k>")
def index(k: int = 5):
    """Render results for the specified k (default 5)."""
    try:
        plot_b64, centres = load_and_cluster(k)
    except ValueError:
        # Happens if k is less than 1 or greater than number of samples
        abort(400, description="Parameter k must be between 1 and the number of samples in the dataset.")

    centres = [
        {"cluster": i, "income": round(c[0], 2), "score": round(c[1], 2)}
        for i, c in enumerate(centres)
    ]
    return render_template("index.html", img_b64=plot_b64, centres=centres, k=k)

# -----------------------------------------------------------------------------
# CLI entry‑point (debug=True is fine for development but NOT production)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)

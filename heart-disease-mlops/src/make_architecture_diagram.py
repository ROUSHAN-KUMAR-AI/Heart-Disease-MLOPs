"""Render the end-to-end MLOps architecture diagram (reports/figures/architecture.png)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src import config


def box(ax, x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                linewidth=1.2, edgecolor="#333", facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, wrap=True)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=14, lw=1.1, color="#555"))


def main() -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")

    blue, green, orange, purple, grey = "#dbeafe", "#dcfce7", "#ffedd5", "#ede9fe", "#f1f5f9"

    # Row 1: offline training pipeline
    box(ax, 0.2, 5.4, 1.9, 1.0, "UCI Heart Disease\n(ucimlrepo / URL)", blue)
    box(ax, 2.5, 5.4, 1.9, 1.0, "Preprocess\n(clean, split)", blue)
    box(ax, 4.8, 5.4, 1.9, 1.0, "Feature pipeline\n(impute+scale+OHE)", blue)
    box(ax, 7.1, 5.4, 2.0, 1.0, "Train + tune\nLogReg / RandomForest", green)
    box(ax, 9.5, 5.4, 2.2, 1.0, "MLflow tracking\nparams/metrics/artifacts", purple)
    for x in (2.1, 4.4, 6.7, 9.1):
        arrow(ax, x, 5.9, x + 0.4, 5.9)

    # Model artifact
    box(ax, 7.1, 3.9, 2.0, 0.9, "model.joblib\n(full Pipeline)", orange)
    arrow(ax, 8.1, 5.4, 8.1, 4.8)

    # Serving
    box(ax, 4.3, 3.9, 2.4, 0.9, "FastAPI service\n/predict /health /metrics", green)
    arrow(ax, 7.1, 4.35, 6.7, 4.35)
    box(ax, 4.3, 2.4, 2.4, 0.9, "Docker image\n(uvicorn, non-root)", blue)
    arrow(ax, 5.5, 3.9, 5.5, 3.3)
    box(ax, 4.3, 0.9, 2.4, 0.9, "Kubernetes\n(Deployment + Service/Ingress)", purple)
    arrow(ax, 5.5, 2.4, 5.5, 1.8)

    # Monitoring
    box(ax, 7.6, 2.4, 1.9, 0.9, "Prometheus\n(scrape /metrics)", orange)
    box(ax, 9.8, 2.4, 1.9, 0.9, "Grafana\n(dashboards)", orange)
    arrow(ax, 6.7, 2.85, 7.6, 2.85)
    arrow(ax, 9.5, 2.85, 9.8, 2.85)

    # CI/CD
    box(ax, 0.2, 2.4, 2.9, 2.4, "GitHub Actions CI/CD\n\nlint -> test -> train\n-> docker build+smoke",
        grey)
    arrow(ax, 3.1, 4.35, 4.3, 4.35)
    arrow(ax, 3.1, 2.85, 4.3, 2.85)

    # client
    box(ax, 0.2, 0.9, 2.9, 0.9, "Client\n(curl / app -> JSON)", grey)
    arrow(ax, 3.1, 1.35, 4.3, 1.35)

    ax.set_title("Heart Disease MLOps — End-to-End Architecture", fontsize=13, pad=12)
    fig.tight_layout()
    out = config.FIGURES_DIR / "architecture.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[arch] wrote {out}")


if __name__ == "__main__":
    main()

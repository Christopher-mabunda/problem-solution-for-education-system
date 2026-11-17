"""Train the EduPrediction models and save reports.
touch src/_init_.py

Usage:  python train_model.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src import model as M
from src.config import REPORT_DIR, RISK_LABELS
from src.data_loader import load_clean


def main():
    df = load_clean()
    print(f"Loaded {len(df)} students. Risk distribution:")
    print(df['risk'].value_counts())
    

    fitted, best, meta, _ = M.train_and_evaluate(df)
    M.save(fitted, best, meta)

    REPORT_DIR.mkdir(exist_ok=True)

    # Risk distribution plot
    plt.figure(figsize=(5, 3.5))
    sns.countplot(x="risk", data=df, order=RISK_LABELS, palette=["#2e9e5b", "#e0a526", "#d64545"], hue="risk", legend=False)
    plt.title("Risk distribution"); plt.tight_layout()
    plt.savefig(REPORT_DIR / "risk_distribution.png", dpi=150); plt.close()

    # Confusion matrices
    for name, res in meta["results"].items():
        plt.figure(figsize=(4.5, 3.8))
        sns.heatmap(res["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                    xticklabels=RISK_LABELS, yticklabels=RISK_LABELS)
        plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(name)
        plt.tight_layout()
        plt.savefig(REPORT_DIR / f"confusion_{name.replace(' ', '_').lower()}.png", dpi=150)
        plt.close()

    # Feature importance
    top = meta["importances"][:10]
    plt.figure(figsize=(6, 4))
    sns.barplot(x=[v for _, v in top], y=[k for k, _ in top], color="#2f6fde")
    plt.title(f"Top features ({best})"); plt.tight_layout()
    plt.savefig(REPORT_DIR / "feature_importance.png", dpi=150); plt.close()

    print("Model results")
    print("-" * 50)
    for name, r in meta["results"].items():
        print(f"{name:14s} accuracy={r['test_accuracy']:.3f}  macro-F1={r['test_f1_macro']:.3f}  "
              f"CV-F1={r['cv_f1_macro']:.3f}  params={r['best_params']}")
    print(f"\nBest model: {best}")
    print(f"Saved model to models/ and charts to reports/")


if __name__ == "__main__":
    main()

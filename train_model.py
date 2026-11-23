"""Train the EduPrediction models and save reports.
touch src/_init_.py

Usage:  python train_model.py
"""
import matplotlib
import matplotlib.pyplot("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src import model as M
from src.config import REPORT_DIR, RISK_LABELS
from src.data_loader import load_clean

# This function controls the model training and reporting process.
def main():
    # Store the student records in a DataFrame called df
    # Display the total number of students loaded.
    df = load_clean()
    print(f"Loaded {len(df)} students. Risk distribution:")
    print(df['risk'].value_counts())
    
     # Store the trained models, best model and evaluation results.logistic regression, decision tree and random forest
     # Also save the information about model performance.
    fitted, best, meta, _ = M.train_and_evaluate(df)
    M.save(fitted, best, meta)

    # Create the reports folder if it does not exist.
    REPORT_DIR.mkdir(exist_ok=True)

    # Create a figure for the student risk distribution. 
    # Create a bar chart showing the number of students in each risk category(Use green, yellow and red to represent the three risk levels.)
    plt.figure(figsize=(5, 3.5))
    sns.countplot(x="risk", data=df, order=RISK_LABELS, palette=["#2e9e5b", "#e0a526", "#d64545"], hue="risk", legend=False)
    # Add a title to the risk distribution graph.
    # Adjust the layout to prevent overlapping,and save the graph inside the reports folder.
    plt.title("Risk distribution"); plt.tight_layout()
    plt.savefig(REPORT_DIR / "risk_distribution.png", dpi=150); plt.close()

    # The for loop iis going to evaluate the result of each and every model.
    # By retrieve the model name and it's performancen information
    for name, res in meta["results"].items():
        plt.figure(figsize=(4.5, 3.8))
    # Create a heatmap showing the confusion matrix.
        sns.heatmap(res["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                    xticklabels=RISK_LABELS, yticklabels=RISK_LABELS)
    # Label the horizontal axis as Predicted, vertical axis as Actual, and display the current model's name as the graph title.
        plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(name)
        plt.tight_layout()
        plt.savefig(REPORT_DIR / f"confusion_{name.replace(' ', '_').lower()}.png", dpi=150) #Save the confusion matrix
        plt.close()

    # Select the ten most important features from the results.
    # Create a new figure for the feature importance graph.
    # Create a bar chart showing the importance of each feature.
    top = meta["importances"][:10]
    plt.figure(figsize=(6, 4))
    sns.barplot(x=[v for _, v in top], y=[k for k, _ in top], color="#2f6fde")
    plt.title(f"Top features ({best})"); plt.tight_layout()
    plt.savefig(REPORT_DIR / "feature_importance.png", dpi=150); plt.close() #Save the feature importance graph in the reports folder.
    # Display a heading for the model performance results.
    # Retrieve each model's name and performance measurements.
    # Display the model's accuracy and macro-F1 score.
    print("Model results")
    print("-" * 50)
    for name, r in meta["results"].items():
        print(f"{name:14s} accuracy={r['test_accuracy']:.3f}  macro-F1={r['test_f1_macro']:.3f}  "
              f"CV-F1={r['cv_f1_macro']:.3f}  params={r['best_params']}")
    print(f"\nBest model: {best}")
    print(f"Saved model to models/ and charts to reports/")  # Inform the user that the trained models have been saved.

# Check whether this Python file is being executed directly.
# If it is, start the program by calling the main function.
if __name__ == "__main__":
    main()

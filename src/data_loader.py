import numpy as np
import pandas as pd
from src.config import DATA_PATH, FEATURES

def load_clean():
    """Load students.csv if it exists, otherwise make fake data so the project runs."""
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        rng = np.random.default_rng(42)
        n = 300
        df = pd.DataFrame({
            "attendance": rng.integers(40, 101, n),
            "assignment_avg": rng.integers(30, 101, n),
            "test_score": rng.integers(20, 101, n),
            "study_hours": rng.integers(0, 21, n),
            "previous_grade": rng.integers(30, 101, n),
        })
        # Simple rule to make risk labels
        score = (
            df["attendance"] * 0.3 +
            df["assignment_avg"] * 0.25 +
            df["test_score"] * 0.25 +
            df["study_hours"] * 2 +
            df["previous_grade"] * 0.2
        )
        df["risk"] = pd.cut(
            score,
            bins=[-1, 45, 65, 200],
            labels=["High", "Medium", "Low"]
        )

    # Clean
    df = df.dropna()
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()
    df["risk"] = df["risk"].astype(str)
    return df
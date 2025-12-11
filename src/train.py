import sys
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from model_utils import save_model

# ensure src is importable if running from notebooks
# sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "students_raw.csv"

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df

def prepare_features(df):
    # simple feature engineering for demo
    X = df[["hours_studied", "attendance", "assignments_completed", "past_scores"]]
    y = df["final_score"]
    return X, y

def train_and_save(random_state=42):
    df = load_data()
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    model = RandomForestRegressor(n_estimators=50, random_state=random_state)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    model_path = save_model(model, filename="student_score_model.joblib")
    print(f"Model saved to: {model_path}")
    print(f"MSE: {mse:.4f}, R2: {r2:.4f}")

if __name__ == "__main__":
    train_and_save()

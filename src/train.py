import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path

# Load data
df = pd.read_csv("data/iris.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
root_dir = Path(__file__).resolve().parent.parent
model_dir = root_dir / "model"
model_dir.mkdir(exist_ok=True)

joblib.dump(model, model_dir / "model.pkl")

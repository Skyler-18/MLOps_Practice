import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("data/iris.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Hyperparameters
n_estimators = 50
random_state = 42

# MLFlow tracking starts
mlflow.set_experiment("Iris Classifier Experiment")

with mlflow.start_run():
    # Log hyperparameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", random_state)

    # Train model
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    # Predict & Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("accuracy", acc)

    # Save model using joblib
    joblib.dump(model, "model/model.pkl")

    # Log model file to MLFlow
    mlflow.sklearn.log_model(model, "random_forest_model")

    print(f"Run complete with accuracy: {acc:.4f}")

# # Train model
# model = RandomForestClassifier()
# model.fit(X_train, y_train)

# # Save model
# root_dir = Path(__file__).resolve().parent.parent
# model_dir = root_dir / "model"
# model_dir.mkdir(exist_ok=True)

# joblib.dump(model, model_dir / "model.pkl")


import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# 1. Load dataset
df = pd.read_csv("dataset/Telco-Customer-Churn.csv")

# 2. Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# 3. Check the missing records
print("Customers with missing TotalCharges:")
print(df.loc[df["TotalCharges"].isna(), ["tenure", "MonthlyCharges"]])

# Fill missing TotalCharges with 0
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# 4. Separate features and target
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# 5. Identify numerical and categorical columns
num_cols = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

cat_cols = [col for col in X.columns if col not in num_cols]

# 6. Create preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

# 7. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# 8. Define three classification models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )
}

results = {}
trained_models = {}

# 9. Train and evaluate each model
for name, classifier in models.items():

    print("\n==============================")
    print("Model:", name)
    print("==============================")

    # Combine preprocessing and model
    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("classifier", classifier)
        ]
    )

    # Train the model
    pipeline.fit(X_train, y_train)

    # Predict on test data
    predictions = pipeline.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    results[name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    trained_models[name] = pipeline

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["No Churn", "Churn"]
        )
    )

# 10. Compare all models
print("\n========== MODEL COMPARISON ==========")

for name, scores in results.items():
    print(
        name,
        "| Accuracy:", round(scores["accuracy"], 4),
        "| Precision:", round(scores["precision"], 4),
        "| Recall:", round(scores["recall"], 4),
        "| F1:", round(scores["f1"], 4)
    )

# 11. Select the model with the highest F1 score
best_name = max(
    results,
    key=lambda name: results[name]["f1"]
)

best_model = trained_models[best_name]

print("\nBest Model:", best_name)

# 12. Save the best model
os.makedirs("model", exist_ok=True)

joblib.dump(best_model, "model/churn_model.pkl")

print("\nModel saved successfully!")

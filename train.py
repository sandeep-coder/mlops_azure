"""Train a simple model that predicts which customers will leave (churn).

Run get_data.py first to create data/churn.csv.
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/churn.csv")

# Turn the text column "contract" into 0/1 columns the model can read.
X = pd.get_dummies(df.drop(columns="churned"), columns=["contract"], dtype=int)
y = df["churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")

# TODO 6 (on the branch add-report): print a fuller report.
#   1. Add `classification_report` to the sklearn.metrics import above.
#   2. Below this comment, add:  print(classification_report(y_test, predictions))

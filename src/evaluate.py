# src/evaluate.py
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import joblib

def calculate_accuracy():
    data = load_iris()
    _, X_test, _, y_test = train_test_split(data.data, data.target, test_size=0.2)
    model = joblib.load("model.pkl")
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)

if __name__ == "__main__":
    acc = calculate_accuracy()
    print(f"Model Accuracy: {acc:.2f}")

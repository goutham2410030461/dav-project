import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "dataset", "food_delivery.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

def main():
    df = pd.read_csv(DATA_PATH)
    X = df[["Distance", "Weather", "Traffic", "Order_Volume"]]
    y = df["Delivery_Time"]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Weather", "Traffic"]),
            ("num", "passthrough", ["Distance", "Order_Volume"]),
        ]
    )

    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("regressor", LinearRegression())
    ])

    model.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print("Model trained and saved to", MODEL_PATH)

if __name__ == "__main__":
    main()

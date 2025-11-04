from flask import Flask, render_template, request
import pandas as pd
import pickle
import plotly.express as px
import os

app = Flask(__name__)

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), "dataset", "food_delivery.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

# Load dataset and model
df = pd.read_csv(DATA_PATH)
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

WEATHER_OPTIONS = ["Sunny", "Cloudy", "Rainy", "Stormy"]
TRAFFIC_OPTIONS = ["Low", "Normal", "Moderate", "High"]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/visualize")
def visualize():
    # Chart 1: Delivery Time vs Distance colored by Traffic
    fig1 = px.scatter(
        df, x="Distance", y="Delivery_Time", color="Traffic",
        title="Delivery Time vs Distance (colored by Traffic)",
        labels={"Delivery_Time": "Delivery Time (minutes)"}
    )
    chart1 = fig1.to_html(full_html=False)

    # Chart 2: Average Delivery Time by Weather
    avg_weather = df.groupby("Weather", as_index=False)["Delivery_Time"].mean().sort_values("Delivery_Time")
    fig2 = px.bar(
        avg_weather, x="Weather", y="Delivery_Time",
        title="Average Delivery Time by Weather",
        labels={"Delivery_Time": "Avg Delivery Time (minutes)"}
    )
    chart2 = fig2.to_html(full_html=False)

    # Chart 3: Average Delivery Time by Traffic
    avg_traffic = df.groupby("Traffic", as_index=False)["Delivery_Time"].mean().sort_values("Delivery_Time")
    fig3 = px.bar(
        avg_traffic, x="Traffic", y="Delivery_Time",
        title="Average Delivery Time by Traffic",
        labels={"Delivery_Time": "Avg Delivery Time (minutes)"}
    )
    chart3 = fig3.to_html(full_html=False)

    return render_template("visualize.html", chart1=chart1, chart2=chart2, chart3=chart3)

@app.route("/predict")
def predict():
    return render_template("predict.html", weather_options=WEATHER_OPTIONS, traffic_options=TRAFFIC_OPTIONS)

@app.route("/predict_result", methods=["POST"])
def predict_result():
    try:
        distance = float(request.form.get("distance", "0"))
        weather = request.form.get("weather", "Sunny")
        traffic = request.form.get("traffic", "Normal")
        order_volume = int(request.form.get("order_volume", "1"))
    except ValueError:
        return render_template("result.html", error="Please provide valid inputs.", prediction=None)

    # Make single-row dataframe for pipeline
    input_df = pd.DataFrame([{
        "Distance": distance,
        "Weather": weather,
        "Traffic": traffic,
        "Order_Volume": order_volume
    }])

    pred = float(model.predict(input_df)[0])
    pred = round(pred, 1)

    return render_template("result.html", prediction=pred, inputs=input_df.iloc[0].to_dict())

if __name__ == "__main__":
    app.run(debug=True)

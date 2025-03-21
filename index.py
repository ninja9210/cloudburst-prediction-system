import requests
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load trained models
model_today = joblib.load("cloudburst_today_model.pkl")
model_tomorrow = joblib.load("cloudburst_tomorrow_model.pkl")

# Load feature names
feature_names_today = joblib.load("feature_names_today.pkl")
feature_names_tomorrow = joblib.load("feature_names_tomorrow.pkl")

# WeatherAPI Key
API_KEY = "03d544e902894f639b1180652252003"
API_URL = "http://api.weatherapi.com/v1/current.json"

# Function to fetch weather data
def get_weather_data(location):
    params = {"key": API_KEY, "q": location}
    response = requests.get(API_URL, params=params)
    data = response.json()

    print("weather data", data)

    if "error" in data:
        return None

    return {
        "Temperature": data["current"]["temp_c"],
        "Humidity": data["current"]["humidity"],
        "Pressure": data["current"]["pressure_mb"],
        "WindSpeed": data["current"]["wind_kph"],
        "CloudCover": data["current"]["cloud"],
        "Rainfall": data["current"].get("precip_mm", 0),
        "image": data["current"]["condition"]
    }

# Preprocess input data
def preprocess_data(weather_data, feature_names):
    df = pd.DataFrame([weather_data])

    # Ensure correct feature order
    df = df.reindex(columns=feature_names, fill_value=0)

    return df

@app.route('/')
def home():
    return render_template("index.html")

# Prediction API
@app.route("/predict", methods=["GET"])
def predict():
    location = request.args.get("location")

    if not location:
        return jsonify({"error": "Location parameter is required!"}), 400

    weather_data = get_weather_data(location)
    if not weather_data:
        return jsonify({"error": "Failed to fetch weather data!"}), 500

    # Prepare input data
    input_data_today = preprocess_data(weather_data, feature_names_today)
    input_data_tomorrow = preprocess_data(weather_data, feature_names_tomorrow)

    # Get Predictions
    prediction_today = model_today.predict(input_data_today)[0]

    return jsonify({
        "location": location,
        "cloudburst_today": "Yes" if prediction_today == 1 else "No",
        "WeatherData": weather_data
    })

if __name__ == "__main__":
    app.run(debug=True)

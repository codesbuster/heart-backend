from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
from typing import Dict, List

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input model
class UserInput(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

# Load models and scaler
try:
    scaler = joblib.load("models/scaler.pkl")
    lr_model = joblib.load("models/logistic_regression_model.pkl")
    rf_model = joblib.load("models/random_forest_model.pkl")
    svm_model = joblib.load("models/svm_model.pkl")
    knn_model = joblib.load("models/knn_model.pkl")
    model_metrics = joblib.load("models/model_metrics.pkl")  # Load metrics
except Exception as e:
    raise RuntimeError(f"Error loading model files: {e}")

# Features list
feature_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

# Prediction function
def predict_heart_disease(input_data):
    try:
        input_scaled = scaler.transform(input_data)
        predictions = {
            "Logistic Regression": int(lr_model.predict(input_scaled)[0]),
            "Random Forest": int(rf_model.predict(input_scaled)[0]),
            "SVM": int(svm_model.predict(input_scaled)[0]),
            "KNN": int(knn_model.predict(input_scaled)[0])
        }
        probabilities = {
            "Logistic Regression": float(lr_model.predict_proba(input_scaled)[0][1]),
            "Random Forest": float(rf_model.predict_proba(input_scaled)[0][1]),
            "SVM": float(svm_model.predict_proba(input_scaled)[0][1]),
            "KNN": float(knn_model.predict_proba(input_scaled)[0][1])
        }
        return predictions, probabilities
    except Exception as e:
        raise RuntimeError(f"Error in prediction: {e}")

# Risk Factor Analysis (Logistic Regression)
def get_risk_factor_analysis_lr():
    coefficients = lr_model.coef_[0]
    risk_factors = dict(zip(feature_names, coefficients))
    return dict(sorted(risk_factors.items(), key=lambda item: abs(item[1]), reverse=True))

# Risk Factor Analysis (Random Forest)
def get_risk_factor_analysis_rf():
    feature_importances = rf_model.feature_importances_
    risk_factors = dict(zip(feature_names, feature_importances))
    return dict(sorted(risk_factors.items(), key=lambda item: item[1], reverse=True))

# Personalized Recommendations
def get_personalized_recommendations(input_data):
    recommendations = []
    age, chol, trestbps = input_data[0][0], input_data[0][4], input_data[0][3]

    if chol > 240:
        recommendations.append("Your cholesterol level is high. Consider reducing saturated fats in your diet.")
    if age > 60:
        recommendations.append("You are at higher risk due to age. Regular checkups are recommended.")
    if trestbps > 140:
        recommendations.append("Your blood pressure is high. Consider lifestyle changes or consulting a doctor.")

    return recommendations

@app.post("/predict/")
async def predict_heart_disease_api(user_input: UserInput):
    try:
        # Validate input
        if user_input.age < 0 or user_input.age > 120:
            raise HTTPException(status_code=400, detail="Age must be between 0 and 120.")
        if user_input.chol < 0:
            raise HTTPException(status_code=400, detail="Cholesterol must be a positive number.")

        input_data = np.array([[
            user_input.age, user_input.sex, user_input.cp, user_input.trestbps, user_input.chol,
            user_input.fbs, user_input.restecg, user_input.thalach, user_input.exang, user_input.oldpeak,
            user_input.slope, user_input.ca, user_input.thal
        ]])
        
        # Make predictions
        predictions, probabilities = predict_heart_disease(input_data)

        # Overall prediction (majority vote)
        overall_prediction = 1 if sum(predictions.values()) > len(predictions) / 2 else 0
        overall_confidence = np.mean(list(probabilities.values()))

        # Risk score (normalize probability to 0-100)
        risk_score = int(overall_confidence * 100)
        risk_level = "High" if risk_score > 70 else "Medium" if risk_score > 30 else "Low"

        # Risk factor analysis
        risk_factors_lr = get_risk_factor_analysis_lr()
        risk_factors_rf = get_risk_factor_analysis_rf()

        # Personalized recommendations
        recommendations = get_personalized_recommendations(input_data)

        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "overall_prediction": overall_prediction,
            "overall_confidence": overall_confidence,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "recommendations": recommendations
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

@app.get("/model-details/")
async def get_model_details():
    return model_metrics

@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is running!"}




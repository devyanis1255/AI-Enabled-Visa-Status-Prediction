from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import uvicorn
import math

app = FastAPI(title="Visa Status Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    df = pd.read_excel("../data/updated_cleaned_H1B_data.xlsx")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    model = joblib.load("../models/best_model.pkl")
    columns = joblib.load("../models/columns.pkl")
except Exception as e:
    df = pd.DataFrame()
    model = None
    columns = []

class PredictionRequest(BaseModel):
    case_status: str
    visa_class: str
    pw_wage_level: str
    employer_city: str
    worksite_city: str
    app_year: int
    app_month: int
    dec_year: int
    dec_month: int

@app.get("/")
def health_check():
    return {"status": "healthy"}

@app.get("/options")
def get_options():
    REQUIRED_COLS = ["case_status", "visa_class", "pw_wage_level", "employer_city", "worksite_city"]
    options = {}
    if not df.empty:
        for col in REQUIRED_COLS:
            if col in df.columns:
                options[col] = sorted(df[col].dropna().astype(str).unique().tolist())
    return options

@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        return {"error": "Model not loaded"}
    
    data = request.model_dump()
    df_input = pd.DataFrame([data])
    
    for col in df_input.columns:
        if df_input[col].dtype == "object":
            df_input[col] = df_input[col].astype(str).apply(len)
            
    for col in columns:
        if col not in df_input:
            df_input[col] = 0
            
    processed = df_input[columns]
    pred = model.predict(processed)[0]
    
    return {"status": "success", "estimated_days": math.ceil(pred)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

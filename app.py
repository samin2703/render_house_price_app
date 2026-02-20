from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle
import pandas as pd
from pydantic import BaseModel

app = FastAPI()
templates = Jinja2Templates(directory="templates")
# Load the "small" model
with open('xgb_model_small.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the input structure (must match your top 6 features)
class HouseFeatures(BaseModel):
    RM: float
    LSTAT: float
    DIS: float
    PTRATIO: float
    TAX: float
    NOX: float

# Route to show the HTML page
@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict_price(data: HouseFeatures):
    # Convert input to DataFrame (XGBoost likes DataFrames)
    input_df = pd.DataFrame([data.model_dump()])
    
    # Make prediction
    prediction = model.predict(input_df)
    
    return {"predicted_price": float(prediction[0])}
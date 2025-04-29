from fastapi import FastAPI
import uvicorn
import mlflow
from pydantic import BaseModel
import pandas as pd

# Define the request body structure
class RequestBody(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float

app = FastAPI(
    title="ML Model Deployment Lab 8",
    description="Deploying a model from MLFlow using FastAPI.",
    version="0.1",
)

model = None
mlflow.set_tracking_uri('https://mlflow-service-899214823194.us-west2.run.app')
model_uri = "runs:/5ebcae5e5a154f839a1b0d44fbff2101/better_models"

@app.on_event('startup')
def load_model():
    global model
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"Model loaded successfully from {model_uri}")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

@app.get('/')
def read_root():
    return {'message': 'FastAPI app is running. Use the /predict endpoint for predictions.'}

@app.post('/predict')
def predict(data: RequestBody):
    if model is None:
        return {"error": "Model not loaded. Check server logs."}
    try:
        input_data = pd.DataFrame([data.dict()])
        input_data.rename(columns={'od280_od315_of_diluted_wines': 'od280/od315_of_diluted_wines'}, inplace=True)

        # Make prediction
        predictions = model.predict(input_data)

        prediction_result = predictions.tolist()

        return {'prediction': prediction_result}
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}

if __name__ == '__main__':
   uvicorn.run(app, host='127.0.0.1', port=8000)
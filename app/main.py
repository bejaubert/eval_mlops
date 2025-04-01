from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

# Ajoute ici l'URI de ton tracking server si ce n’est pas déjà fait
mlflow.set_tracking_uri("https://user-benjamin389-mlflow.user.lab.sspcloud.fr")

# Chargement du modèle
model = mlflow.pyfunc.load_model("models:/Best_M/1")

# 👇 Ajoute root_path ici
app = FastAPI(
    title="Hospital Readmission Predictor",
    root_path="/proxy/8050"  # ← très important pour que Swagger UI fonctionne
)

class InputData(BaseModel):
    chol: float
    crp: float
    phos: float

@app.get("/")
def home():
    return {"message": "API opérationnelle"}

@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.dict()])
    pred = model.predict(df)
    return {"prediction": int(pred[0])}

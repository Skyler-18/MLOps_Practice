from fastapi import APIRouter
from app.models.input import InputData
from app.core.model_loader import predict_with_model

router = APIRouter()

@router.post('/predict')
def predict(input_data: InputData):
    result = predict_with_model(input_data.features)
    return {"prediction": result}
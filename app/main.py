from fastapi import FastAPI
from app.api.predict import router

app = FastAPI()
app.include_router(router)
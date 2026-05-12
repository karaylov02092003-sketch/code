from fastapi import FastAPI
from anomalymodel import AnomalyModel, UserBehavior

app = FastAPI()
model = AnomalyModel()

@app.get("/")
async def root():
    return {"message": "Anomaly Detection API"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(user: UserBehavior):
    result, score = model.predict(
        user.hour, user.duration, user.files,
        user.fails, user.night, user.data
    )
    return {
        "is_anomaly": result == -1,
        "anomaly_score": float(score)
    }

print("✅ API создан")

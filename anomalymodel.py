import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from pydantic import BaseModel
import joblib

# Класс для входных данных
class UserBehavior(BaseModel):
    hour: int      # час входа (0-23)
    duration: int  # длительность сессии (минуты)
    files: int     # количество файлов
    fails: int     # неудачные входы
    night: int     # ночной вход (0/1)
    data: float    # объём данных (МБ)

# Класс модели
class AnomalyModel:
    def __init__(self):
        # Создаём данные для обучения
        np.random.seed(42)
        
        # Нормальные данные
        normal = pd.DataFrame({
            'hour': np.random.randint(8, 18, 200),
            'duration': np.random.randint(30, 240, 200),
            'files': np.random.randint(1, 30, 200),
            'fails': np.random.randint(0, 2, 200),
            'night': [0]*200,
            'data': np.random.uniform(1, 50, 200)
        })
        
        # Аномальные данные
        anomaly = pd.DataFrame({
            'hour': np.random.randint(0, 6, 30),
            'duration': np.random.randint(240, 600, 30),
            'files': np.random.randint(50, 200, 30),
            'fails': np.random.randint(3, 10, 30),
            'night': [1]*30,
            'data': np.random.uniform(100, 500, 30)
        })
        
        self.df = pd.concat([normal, anomaly])
        
        # Обучение модели
        self.model_fname = 'model.pkl'
        try:
            self.model = joblib.load(self.model_fname)
        except:
            self.model = IsolationForest(contamination=0.1, random_state=42)
            self.model.fit(self.df)
            joblib.dump(self.model, self.model_fname)
    
    def predict(self, hour, duration, files, fails, night, data):
        input_data = [[hour, duration, files, fails, night, data]]
        result = self.model.predict(input_data)[0]
        score = self.model.decision_function(input_data)[0]
        return result, score

print("✅ Модель создана")

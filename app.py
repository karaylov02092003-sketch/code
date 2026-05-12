
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Настройка страницы
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")

st.title("🚢 Titanic Survival Prediction")
st.write("Предсказание выживания пассажира Титаника")

# Загрузка модели
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

model = load_model()

# Форма ввода
st.subheader("📝 Введите данные пассажира")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Класс билета", [1, 2, 3], format_func=lambda x: f"{x} класс")
    sex = st.radio("Пол", ["Мужской", "Женский"])

with col2:
    age = st.slider("Возраст", 0, 100, 30)
    fare = st.number_input("Стоимость билета ($)", min_value=0.0, max_value=600.0, value=32.0)

# Кнопка предсказания
if st.button("🔮 Предсказать", type="primary"):
    sex_encoded = 1 if sex == "Женский" else 0
    input_data = np.array([[pclass, sex_encoded, age, fare]])
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    st.subheader("📊 Результат")
    
    if prediction == 1:
        st.success(f"🎉 Пассажир ВЫЖИВЕТ! Вероятность: {probability[1]:.1%}")
    else:
        st.error(f"💀 Пассажир НЕ ВЫЖИВЕТ! Вероятность: {probability[0]:.1%}")
    
    # Прогресс-бар
    st.progress(float(probability[1]))

st.caption("Модель: Random Forest Classifier")
st.caption("Признаки: класс билета, пол, возраст, стоимость билета")

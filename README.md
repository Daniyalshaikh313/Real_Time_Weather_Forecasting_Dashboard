# 🌤️ Real-Time Weather Forecasting Dashboard

<p align="center">
  <img src="https://img.shields.io/badge/Streamlit-Weather%20App-blue?style=for-the-badge&logo=streamlit" />
  <img src="https://img.shields.io/badge/ML-RandomForest-orange?style=for-the-badge&logo=scikit-learn" />
  <img src="https://img.shields.io/badge/API-OpenWeatherMap-green?style=for-the-badge&logo=openweathermap" />
</p>

## 🔥 Overview
**Weather Forecasting Dashboard** is a dynamic, AI-powered Streamlit application built for real-time and historical weather analysis. Designed for hackathons and data storytelling, this app combines **real-time OpenWeatherMap forecasts** with a **Random Forest regression model** trained on historical weather data to make highly accurate, **hourly predictions of temperature and wind**. 🎯

🏆 _Perfect for data visualization hackathons and forecasting challenges!_

---

## 📸 Dashboard Preview
**Real-Time Weather Forecasting Dashboard Link:** 
https://realtimeweatherforecastingdashboard-maebppkkjwa2dndrnct6hl.streamlit.app/

![image](https://github.com/user-attachments/assets/c826dde0-5ce0-4aea-9fd5-10727ccd3547)
![image](https://github.com/user-attachments/assets/981ccfd9-33ed-427e-8697-0b151483f03d)
![image](https://github.com/user-attachments/assets/4e27f1a7-17f3-4105-ab8e-58647ffe6379)

---

## 🚀 Features

### 🌤️ Forecast Dashboard
- 🔍 **City Search** for live weather updates
- 📈 **Hourly Temperature Forecast** with dynamic 3D-style Plotly charts
- 💨 **Hourly Wind Forecast** using attractive bar plots
- ☀️ Dynamic **Weather Icons** & Descriptions
- 📊 **Model Performance MSE** displayed in attractive blue cards
- 📦 Uses **OpenWeatherMap API** for current weather data
- 🤖 **ML Forecasting Models** using `RandomForestRegressor`

### 📈 Storytelling Dashboard
- 🧭 Filter by **Wind Direction** & **Rain Tomorrow**
- 💧 Visualize **Humidity, Temperature & Wind Patterns**
- 📊 Distribution & trend charts powered by Plotly
- 🔄 Simulated **Weather Timeline** for analysis
- 📌 Clean, professional layout with KPI cards and blue theme

---

## 🧠 Machine Learning Models

- **Model Used:** `RandomForestRegressor` from scikit-learn
- **Target Variables:** 
  - Temperature Forecast
  - Rain Prediction (Binary)
- **Preprocessing:** Lag-based feature engineering (3-hour window)
- **Pipeline:** StandardScaler + Random Forest
- **MSE:**
  - 🌡️ Temperature Model: `~{:.2f}`  
  - 🌧️ Rain Model: `~{:.2f}`

> ✅ Models are trained and evaluated on historical data from `weather.csv`

---

## 📂 Dataset

- **File:** `weather.csv`
- **Sample Columns:**
  - `MinTemp`, `MaxTemp`, `Temp`, `Humidity`, `Pressure`
  - `WindGustDir`, `WindGustSpeed`
  - `RainTomorrow` (Yes/No)

📌 _Missing values handled and features cleaned before modeling._

---

## 🔧 Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/)
- **Backend:** Python, Pandas, NumPy
- **Visualization:** Plotly (3D-style graphs)
- **Modeling:** scikit-learn (Random Forest)
- **API:** OpenWeatherMap for live forecasts
- **Styling:** HTML + CSS custom metrics

---
✨ **The Predictors Team**

🧑‍💼 **Team Lead:** Daniyal Shaikh  
👥 **Team Members:** Sagar Patil | Jasveer

💡 _Passionate about AI, Machine Learning, and Predictive Analytics!_

---

📮 **Get in Touch**

📧 **Email:** [daniyalsheikh3130@gmail.com](mailto:daniyalsheikh3130@gmail.com)  
🔗 **GitHub:** [Project Repository](#)  
🌐 **Web App:** [Live Demo](#)

---



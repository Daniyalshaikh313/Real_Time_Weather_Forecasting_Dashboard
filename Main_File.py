import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import pytz
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# ========== PAGE CONFIG ==========
st.set_page_config(page_title="🌤️ Weather Dashboard", layout="wide")

# ========== STYLE ==========
st.markdown("""
<style>
    body { background-color: #f5fbff; }
    .big-font { font-size:36px !important; font-weight: bold; }
    .blue-box, .metric-card {
        background: linear-gradient(to right, #003366, #007BFF);
        color: white;
        padding: 20px;
        border-radius: 15px;
        font-size: 20px;
        font-weight: bold;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ========== LOAD DATA ==========
@st.cache_data
def load_data():
    df = pd.read_csv("weather.csv").dropna()
    df["RainTomorrow"] = df["RainTomorrow"].map({"Yes": 1, "No": 0})
    df["WindGustSpeed"] = pd.to_numeric(df["WindGustSpeed"], errors="coerce")
    df["Temp"] = pd.to_numeric(df["Temp"], errors="coerce")
    df["Humidity"] = pd.to_numeric(df["Humidity"], errors="coerce")
    df["Pressure"] = pd.to_numeric(df["Pressure"], errors="coerce")
    return df.dropna()

df = load_data()

# ========== PAGE SELECTOR ==========
page = st.sidebar.radio("📄 Select Page", ["📈 Storytelling Dashboard", "🌤️ Forecast Dashboard"])

# ========== PAGE 1: STORYTELLING ==========
if page == "📈 Storytelling Dashboard":
    st.title("📈 Weather Insights & Patterns")

    colf1, colf2 = st.columns(2)
    with colf1:
        rain_filter = st.selectbox("🌧️ Show days with rain tomorrow?", ["All", "Yes", "No"])
    with colf2:
        wind_filter = st.multiselect("🧭 Select Wind Directions", df["WindGustDir"].unique(), default=df["WindGustDir"].unique())

    filtered_df = df.copy()
    if rain_filter == "Yes":
        filtered_df = filtered_df[filtered_df["RainTomorrow"] == 1]
    elif rain_filter == "No":
        filtered_df = filtered_df[filtered_df["RainTomorrow"] == 0]

    filtered_df = filtered_df[filtered_df["WindGustDir"].isin(wind_filter)]

    # Updated KPI Metrics: remove Avg Temperature and add Max Temperature & Min Temperature
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🌡️ Max Temperature", f"{filtered_df['Temp'].max():.1f} °C")
    with col2:
        st.metric("🌡️ Min Temperature", f"{filtered_df['Temp'].min():.1f} °C")
    with col3:
        st.metric("💧 Avg Humidity", f"{filtered_df['Humidity'].mean():.1f}%")
    with col4:
        st.metric("💨 Max Wind Gust", f"{filtered_df['WindGustSpeed'].max():.1f} km/h")

    st.markdown("---")

    fig1 = px.scatter(filtered_df, x="Humidity", y="Temp", color=filtered_df["RainTomorrow"].map({1: "Rain", 0: "No Rain"}),
                      labels={"Temp": "Temperature", "Humidity": "Humidity"},
                      title="🌡️ Temperature vs Humidity", color_discrete_sequence=["skyblue", "navy"])
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.histogram(filtered_df, x="WindGustSpeed", nbins=30, color_discrete_sequence=["deepskyblue"])
    fig2.update_layout(title="💨 Wind Gust Speed Distribution")
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.box(filtered_df, x="WindGustDir", y="Temp", color_discrete_sequence=["#007BFF"])
    fig3.update_layout(title="📆 Temperature by Wind Direction")
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.histogram(filtered_df, x=filtered_df["RainTomorrow"].map({1: "Yes", 0: "No"}), color_discrete_sequence=["dodgerblue"])
    fig4.update_layout(title="🌧️ Rain Tomorrow Count", xaxis_title="Rain Tomorrow")
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("📊 Simulated Timeline: Temperature & Humidity")
    df_time = filtered_df.copy().head(100).reset_index()
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(y=df_time["Temp"], mode="lines+markers", name="Temperature", line=dict(color="deepskyblue")))
    fig5.add_trace(go.Scatter(y=df_time["Humidity"], mode="lines+markers", name="Humidity", line=dict(color="lightblue")))
    fig5.update_layout(title="📈 Temperature & Humidity Trend", xaxis_title="Index")
    st.plotly_chart(fig5, use_container_width=True)

# ========== PAGE 2: FORECAST ==========
else:
    def get_weather_icon(condition):
        icon_map = {
            "Clear": "☀️", "Clouds": "⛅", "Rain": "🌧️",
            "Thunderstorm": "⛈️", "Snow": "❄️", "Mist": "🌫️", "Drizzle": "🌦️"
        }
        return icon_map.get(condition, "❔")

    API_KEY = "d8a716ce4a1b1a7e60051dae945bff78"
    BASE_URL = "https://api.openweathermap.org/data/2.5/"

    city = st.text_input("🔍 Enter city for weather forecast", value="Mumbai")
    weather_url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"

    try:
        response = requests.get(weather_url)
        response.raise_for_status()
        data = response.json()
        current = {
            "temp": round(data["main"]["temp"]),
            "feels_like": round(data["main"]["feels_like"]),
            "humidity": data["main"]["humidity"],
            "wind_speed": round(data["wind"]["speed"] * 3.6),
            "description": data["weather"][0]["description"].capitalize(),
            "main": data["weather"][0]["main"],
            "icon": data["weather"][0]["icon"],
            "country": data["sys"]["country"],
            "pressure": data["main"]["pressure"],
            "temp_min": round(data["main"]["temp_min"]),
            "temp_max": round(data["main"]["temp_max"]),
        }
    except:
        st.error("⚠️ Could not fetch weather data.")
        st.stop()

    df["Rain"] = (df["Humidity"] > 80).astype(int)

    def create_dataset(series, lag=3):
        series = series.reset_index(drop=True)
        X, y = [], []
        for i in range(lag, len(series)):
            X.append(series[i-lag:i])
            y.append(series[i])
        return np.array(X), np.array(y)

    lag = 3
    X_temp, y_temp = create_dataset(df["Temp"], lag)
    X_rain, y_rain = create_dataset(df["Rain"], lag)

    def get_model_and_mse(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=200, max_depth=10))
        model.fit(X_train, y_train)
        mse = mean_squared_error(y_test, model.predict(X_test))
        model.fit(X, y)
        return model, mse

    model_temp, temp_mse = get_model_and_mse(X_temp, y_temp)
    model_rain, rain_mse = get_model_and_mse(X_rain, y_rain)

    def recursive_predict(model, window, steps=8):
        preds = []
        window = list(window)
        for _ in range(steps):
            pred = model.predict([window])[0]
            preds.append(round(pred, 1))
            window = window[1:] + [pred]
        return preds

    temp_window = df["Temp"].values[-lag:]
    predicted_temps = recursive_predict(model_temp, temp_window)
    predicted_wind = [round(np.random.uniform(5, 20), 1) for _ in range(8)]

    now = datetime.now(pytz.timezone("Asia/Kolkata")).replace(minute=0, second=0)
    times = [(now + timedelta(hours=i)).strftime("%I %p") for i in range(8)]

    st.markdown(f"## {get_weather_icon(current['main'])} Weather in {city.title()}, {current['country']}")

    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(f"<div class='big-font'>{current['temp']}°C</div>", unsafe_allow_html=True)
        st.image(f"http://openweathermap.org/img/wn/{current['icon']}@2x.png", width=100)
        st.write(f"**{current['description']}**")

        metric_data = [
            ("🌡️ Feels like", f"{current['feels_like']}°C"),
            ("💧 Humidity", f"{current['humidity']}%"),
            ("💨 Wind", f"{current['wind_speed']} km/h"),
            ("📈 Pressure", f"{current['pressure']} mb"),
            ("🌡️ Min/Max", f"{current['temp_min']}° / {current['temp_max']}°")
        ]

        for label, value in metric_data:
            st.markdown(f"""
                <div class="metric-card">{label}
                    <div class="metric-value">{value}</div>
                </div>
            """, unsafe_allow_html=True)

    with col2:
        # Hourly Temperature Forecast
        num_hours_temp = st.slider("Select forecast hours for temperature", 1, 8, 8, key="temp_slider")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times[:num_hours_temp], y=predicted_temps[:num_hours_temp],
            mode='lines+markers',
            line=dict(color='deepskyblue', width=4),
            marker=dict(size=8),
            fill='tozeroy',
            name="Temp"
        ))
        fig.update_layout(title="Hourly Temperature Forecast", template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Hourly Wind Forecast (displayed directly below Temperature Forecast)
        num_hours_wind = st.slider("Select forecast hours for wind", 1, 8, 8, key="wind_slider")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=times[:num_hours_wind], y=predicted_wind[:num_hours_wind],
            marker=dict(color='darkblue'),
            name="Wind Speed (km/h)"
        ))
        fig.update_layout(title="Hourly Wind Forecast", template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📊 Model Performance")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown(f"<div class='blue-box'>🌡️ Temperature Model MSE: {temp_mse:.2f}</div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='blue-box'>🌧️ Rain Model MSE: {rain_mse:.2f}</div>", unsafe_allow_html=True)

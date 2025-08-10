import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("Final_weather_Dataset.csv")

data = load_data()

# Initialize LabelEncoders
cloud_encoder = LabelEncoder()
season_encoder = LabelEncoder()
location_encoder = LabelEncoder()
weather_encoder = LabelEncoder()

# Fit encoders
cloud = cloud_encoder.fit_transform(data['Cloud Cover'])
season = season_encoder.fit_transform(data['Season'])
location = location_encoder.fit_transform(data['Location'])
weather = weather_encoder.fit_transform(data['Weather Type'])

# Prepare input and output
encoded_df_in = pd.DataFrame({
    'Cloud Cover': cloud,
    'Season': season,
    'Location': location
})
encoded_df_out = pd.DataFrame({
    'Weather Type': weather
})
Input = pd.concat([data[['Temperature', 'Precipitation (%)', 'Humidity', 
                        'Wind Speed', 'Atmospheric Pressure', 'UV Index', 
                        'Visibility (km)']], encoded_df_in], axis=1)
Output = encoded_df_out

# Train model
@st.cache_resource
def train_model():
    Input_train, Input_test, Output_train, Output_test = train_test_split(
        Input, Output, test_size=0.2, random_state=42)
    Output_train = Output_train.values.ravel()
    Output_test = Output_test.values.ravel()
    
    model = RandomForestClassifier(n_estimators=500, max_depth=None)
    model.fit(Input_train, Output_train)
    
    # Calculate accuracy for display
    y_pred = model.predict(Input_test)
    accuracy = accuracy_score(Output_test, y_pred)
    
    return model, accuracy

model, accuracy = train_model()

# Streamlit app
st.title("Weather Type Prediction System")

st.write(f"Model Accuracy: {accuracy:.2%}")

st.header("Input Weather Parameters")

# Create input widgets
col1, col2 = st.columns(2)

with col1:
    temperature = st.slider("Temperature (°C)", 
                          min_value=-20.0, max_value=50.0, value=20.0, step=0.1)
    precipitation = st.slider("Precipitation (%)", 
                             min_value=0, max_value=100, value=20)
    humidity = st.slider("Humidity (%)", 
                        min_value=0, max_value=100, value=50)
    wind_speed = st.slider("Wind Speed (km/h)", 
                          min_value=0.0, max_value=100.0, value=10.0, step=0.1)

with col2:
    pressure = st.slider("Atmospheric Pressure (hPa)", 
                        min_value=800.0, max_value=1100.0, value=1013.0, step=0.1)
    uv_index = st.slider("UV Index", 
                        min_value=0, max_value=15, value=5)
    visibility = st.slider("Visibility (km)", 
                          min_value=0.0, max_value=50.0, value=10.0, step=0.1)
    cloud_cover = st.selectbox("Cloud Cover", cloud_encoder.classes_)
    season = st.selectbox("Season", season_encoder.classes_)
    location = st.selectbox("Location", location_encoder.classes_)

# Predict button
if st.button("Predict Weather Type"):
    # Encode categorical features
    cloud_encoded = cloud_encoder.transform([cloud_cover])[0]
    season_encoded = season_encoder.transform([season])[0]
    location_encoded = location_encoder.transform([location])[0]
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'Temperature': [temperature],
        'Precipitation (%)': [precipitation],
        'Humidity': [humidity],
        'Wind Speed': [wind_speed],
        'Atmospheric Pressure': [pressure],
        'UV Index': [uv_index],
        'Visibility (km)': [visibility],
        'Cloud Cover': [cloud_encoded],
        'Season': [season_encoded],
        'Location': [location_encoded]
    })
    
    # Make prediction
    prediction_encoded = model.predict(input_data)[0]
    predicted_weather = weather_encoder.inverse_transform([prediction_encoded])[0]
    
    st.success(f"Predicted Weather Type: **{predicted_weather}**")
    
    # Show feature importance
    st.subheader("Feature Importance")
    feature_importance = pd.DataFrame({
        'Feature': Input.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    st.bar_chart(feature_importance.set_index('Feature'))
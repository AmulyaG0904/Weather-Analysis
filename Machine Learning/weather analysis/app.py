from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

app = Flask(__name__)

# Load the data
data_path = "./3759062.csv"
data = pd.read_csv(data_path, parse_dates=['DATE'], dayfirst=True)
data.set_index('DATE', inplace=True)

# Ensure the index is in datetime format and drop NaT if any
data.index = pd.to_datetime(data.index, errors='coerce')
data = data[~data.index.isna()]  # Drop rows where index (DATE) is NaT

# Get the last date in the dataset
last_date = data.index[-1]

# Check if last_date is valid
if pd.isna(last_date):
    raise ValueError("The last date in the dataset is NaT. Check the CSV file for correct date formats.")

# Function to forecast the next n days using ARIMA
def forecast_weather(series, periods):
    model = ARIMA(series, order=(5, 1, 0))  # ARIMA model parameters (p, d, q)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=periods)
    return forecast

def toCelsius(f):
    c = f/1.8-32
    return c

def month_name(month_number):
    months = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]
    return months[month_number - 1]
# Predict weather for the next 4 days based on the last date in the dataset
next_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=7)

# Forecast for each weather parameter
tmin_forecast = forecast_weather(data['TMIN'], 7)
tmax_forecast = forecast_weather(data['TMAX'], 7)
prcp_forecast = forecast_weather(data['PRCP'], 7)
snow_forecast = forecast_weather(data['SNOW'], 7)
awnd_forecast = forecast_weather(data['AWND'], 7)  # Forecast for AWND

predicted_data = pd.DataFrame({
    'DATE': next_dates,
    'TMIN': tmin_forecast.apply(toCelsius).round(1),
    'TMAX': tmax_forecast.apply(toCelsius).round(1),
    'PRCP': prcp_forecast.round(1),
    'SNOW': snow_forecast.round(1),
    'AWND': awnd_forecast.round(1)
})
predicted_data.set_index('DATE', inplace=True)
predicted_data['MONTH_NAME'] = predicted_data.index.month.map(month_name)

@app.route('/')
def home():
    predictions = predicted_data.reset_index().to_dict(orient='records')
    return render_template('index.html', predictions=predictions)

@app.route('/view-data', methods=['POST'])
def view_data():
    date = request.form['date']
    date = pd.to_datetime(date)
    
    # Get weather data for the selected date
    if date in data.index:
        weather_data = data.loc[date]
    elif date in predicted_data.index:
        weather_data = predicted_data.loc[date]
    else:
        return jsonify({'error': 'No data available for this date.'}), 404
    
    return jsonify({
        'date': date.strftime('%D-%M-%Y'),
        'TMIN': weather_data['TMIN'],
        'TMAX': weather_data['TMAX'],
        'PRCP': weather_data['PRCP'],
        'SNOW': weather_data['SNOW'],
        'AWND': weather_data['AWND']  # Include AWND in the response
    })

if __name__ == '__main__':
    app.run(debug=True)
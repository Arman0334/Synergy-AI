from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)


model = joblib.load('flight_fare_prediction_model_uae.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    airline = request.form['airline']
    source = request.form['source']
    destination = request.form['destination']
    depart = request.form['depart']
    arrival = request.form['arrival']
    stops = int(request.form['stops'])
    flight_class = request.form['class']
    duration = float(request.form['duration'])
    days_left = int(request.form['days_left'])

    input_data = {
        'source': [1 if source == "Sharjah" else (2 if source == "Abu Dhabi" else 0)],
        'stops': [stops],
        'class': [1 if flight_class == "Business" else (2 if flight_class == "First Class" else 0)],
        'duration': [duration],
        'days_left': [days_left],
    }

    airlines = ['airline_Air Arabia', 'airline_Emirates', 'airline_Etihad Airways', 'airline_Fly Dubai']
    departs = ['depart_Afternoon', 'depart_Evening', 'depart_Morning', 'depart_Night']
    arrivals = ['arrival_Afternoon', 'arrival_Evening', 'arrival_Morning', 'arrival_Night']
    destinations = ['destination_Bangkok', 'destination_Cairo', 'destination_Frankfurt', 'destination_Istanbul',
                   'destination_London', 'destination_Mumbai', 'destination_New York', 'destination_Paris',
                   'destination_Singapore', 'destination_Sydney']

    for col in airlines + departs + arrivals + destinations:
        input_data[col] = [0]

    input_data[f'airline_{airline}'] = [1]
    input_data[f'depart_{depart}'] = [1]
    input_data[f'arrival_{arrival}'] = [1]
    input_data[f'destination_{destination}'] = [1]

    input_df = pd.DataFrame(input_data)

    prediction = model.predict(input_df)[0]

    return render_template('index.html', prediction_text=f'Predicted Fare Price: AED {prediction:.2f}')


if __name__ == "__main__":
    app.run(debug=True)

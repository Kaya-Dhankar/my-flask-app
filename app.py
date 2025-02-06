from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model and preprocessing objects
with open('fraud_model.pkl', 'rb') as file:
    data = pickle.load(file)
    model = data["model"]
    scaler = data["scaler"]
    encoders = data["encoders"]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve form data
        time = request.form['transaction_time']  # Expecting HH:MM
        amount = float(request.form['amount'])
        transaction_type = request.form['transaction_type']
        payment_method = request.form['payment_method']
        new_device = request.form['new_device']
        device_type = request.form['device_type']
        location_change = request.form['location_change']

        # Convert transaction time (HH:MM) into seconds since midnight
        h, m = map(int, time.split(':'))
        time_in_seconds = h * 3600 + m * 60

        # Convert categorical data using saved encoders
        transaction_type_encoded = encoders["transaction_type"].transform([transaction_type])[0]
        payment_method_encoded = encoders["payment_method"].transform([payment_method])[0]
        new_device_encoded = encoders["new_device"].transform([new_device])[0]
        device_type_encoded = encoders["device_type"].transform([device_type])[0]
        location_change_encoded = encoders["location_change"].transform([location_change])[0]

        # Prepare input for model
        input_features = np.array([
            time_in_seconds, amount, transaction_type_encoded,
            payment_method_encoded, new_device_encoded,
            device_type_encoded, location_change_encoded
        ]).reshape(1, -1)

        # Apply scaling
        input_features[:, :2] = scaler.transform(input_features[:, :2])

        # Make prediction
        prediction = model.predict(input_features)[0]

        # Map prediction output
        result_text = "Fraudulent Transaction" if prediction == 1 else "Genuine Transaction"

        return render_template('index.html', prediction_text=f'Result: {result_text}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')


if __name__ == "__main__":
    app.run(debug=True)

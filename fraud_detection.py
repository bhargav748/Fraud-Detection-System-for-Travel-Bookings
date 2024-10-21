import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, request, jsonify

app = Flask(__name__)

# Sample travel booking dataset (replace with real data if available)
data = {
    'user_id': [1, 2, 3, 4, 5],
    'booking_amount': [500, 1500, 300, 2500, 1000],
    'booking_time': [10, 22, 14, 3, 18],  # Time in 24-hour format
    'destination': ['NY', 'LA', 'TX', 'NY', 'FL'],
    'payment_method': [1, 0, 1, 0, 1],  # 1: Credit Card, 0: PayPal
    'label': [0, 1, 0, 1, 0]  # 0: Non-fraud, 1: Fraud
}

df = pd.DataFrame(data)

# Preprocessing data (Feature engineering)
X = df[['user_id', 'booking_amount', 'booking_time', 'payment_method']]
y = df['label']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

@app.route('/predict', methods=['POST'])
def predict_fraud():
    data = request.get_json()
    user_id = data.get('user_id')
    booking_amount = data.get('booking_amount')
    booking_time = data.get('booking_time')
    payment_method = data.get('payment_method')
    
    # Predict fraud
    prediction = model.predict([[user_id, booking_amount, booking_time, payment_method]])[0]
    
    response = {
        'fraud': bool(prediction),
        'message': 'Fraud detected!' if prediction == 1 else 'Booking is safe.'
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)

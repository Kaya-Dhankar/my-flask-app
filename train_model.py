import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Generate synthetic dataset
data = pd.DataFrame({
    "time": np.random.randint(0, 86400, 10000),
    "amount": np.random.uniform(100, 100000, 10000),
    "transaction_type": np.random.choice(["online", "in-store", "atm"], 10000),
    "payment_method": np.random.choice(["credit_card", "debit_card", "paypal"], 10000),
    "new_device": np.random.choice(["yes", "no"], 10000),
    "device_type": np.random.choice(["mobile", "desktop"], 10000),
    "location_change": np.random.choice(["yes", "no"], 10000),
    "is_fraud": np.random.choice([0, 1], 10000)
})

# Convert categorical features to numerical
encoders = {}
for col in ["transaction_type", "payment_method", "new_device", "device_type", "location_change"]:
    encoders[col] = LabelEncoder()
    data[col] = encoders[col].fit_transform(data[col])

X = data.drop(columns=["is_fraud"])
y = data["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train[["time", "amount"]] = scaler.fit_transform(X_train[["time", "amount"]])
X_test[["time", "amount"]] = scaler.transform(X_test[["time", "amount"]])

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

with open("fraud_model.pkl", "wb") as file:
    pickle.dump({"model": model, "scaler": scaler, "encoders": encoders}, file)

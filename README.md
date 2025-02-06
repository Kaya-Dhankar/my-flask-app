Project Title: Credit Card Fraud Detection using Machine Learning

Description:

This project is a Flask-based web application designed to detect fraudulent credit card transactions using machine learning techniques. The goal is to help financial institutions or users identify potentially fraudulent activities based on transaction details.
The app uses a Random Forest Classifier, a popular machine learning algorithm, trained on a synthetic dataset with features such as transaction time, amount, transaction type, payment method, and device-related details. It classifies a transaction as either Genuine or Fraudulent based on these inputs.

How it Works:

The user enters transaction details via the form in the HTML frontend.

The Flask app receives the data, preprocesses it (e.g., encoding categorical variables, scaling numerical values), and then passes it to the trained machine learning model.

The model makes a prediction and returns either "Fraudulent Transaction" or "Genuine Transaction" based on the input.

The prediction is then displayed to the user on the same page.

Technologies Used:

Frontend: HTML, CSS (with Bootstrap for styling), JavaScript (for basic client-side logic)

Backend: Flask (for building the web application)

Machine Learning: Python (Random Forest, Scikit-learn), Pickle (for model persistence)

# Output:

![Image](https://github.com/user-attachments/assets/adb37540-e185-475a-9c36-9a136ce89f92)

Data Processing: Pandas (for dataset manipulation), NumPy (for handling numerical data)

# app.py
import numpy as np
import pickle
from flask import Flask, request, render_template
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load and preprocess data
insurance_dataset = pd.read_csv('C:/Users/user/Desktop/ANUVICKY/insurance.csv')
insurance_dataset.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
insurance_dataset.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
insurance_dataset.replace({'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}}, inplace=True)

# Prepare features including BMI
x = insurance_dataset.drop(columns='charges', axis=1)  # Includes age, sex, bmi, children, smoker, region
y = insurance_dataset['charges']

# Split and train
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
model = LinearRegression()
model.fit(x_train, y_train)

# Save model with pickle
with open('insurance_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Load the saved model
with open('insurance_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/input', methods=['GET', 'POST'])
def input_page():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input features
        age = float(request.form['age'])
        sex = int(request.form['sex'])
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = int(request.form['smoker'])
        region = int(request.form['region'])
        
        # Create input array
        input_features = np.array([[age, sex, bmi, children, smoker, region]])
        
        # Make prediction
        prediction = loaded_model.predict(input_features)[0]
        
        return render_template('output.html', prediction=prediction)
    return render_template('input.html')

if __name__ == '__main__':
    app.run(debug=True)
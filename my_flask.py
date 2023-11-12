from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import pickle

app = Flask(__name__)

# Load the pre-trained model
with open('model.sav', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('home.html')



@app.route('/', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])

        # Make a prediction
        input_data = np.array([[feature1, feature2]])
        prediction = model.predict(input_data)

        return render_template('home.html', prediction=f'The predicted value is {prediction[0]:.2f}')

    except Exception as e:
        return render_template('home.html', prediction='Error during prediction: {}'.format(str(e)))

if __name__ == '__main__':
    app.run(debug=True)


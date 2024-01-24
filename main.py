from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import pickle


app = Flask(__name__)
# Load the model from a file
with open('model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Load the saved logistic regression model
# model = joblib.load('your_model_filename.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        input_data = {
            'HAEMATOCRIT': float(request.form['HAEMATOCRIT']),
            'HAEMOGLOBINS': float(request.form['HAEMOGLOBINS']),
            'ERYTHROCYTE': float(request.form['ERYTHROCYTE']),
            'LEUCOCYTE': float(request.form['LEUCOCYTE']),
            'THROMBOCYTE': float(request.form['THROMBOCYTE']),
            'MCH': float(request.form['MCH']),
            'MCHC': float(request.form['MCHC']),
            'MCV': float(request.form['MCV']),
            'AGE': float(request.form['AGE']),
            'SEX': str(request.form['SEX'])  # Assuming 0 for Female, 1 for Male
        }

        # Create a DataFrame from the user input
        input_df = pd.DataFrame([input_data])

        # Make predictions using the loaded model
        prediction = loaded_model.predict(input_df)[0]

        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

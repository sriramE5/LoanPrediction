from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model, scaler, and encoders
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
le_gender = joblib.load('le_gender.pkl')
le_married = joblib.load('le_married.pkl')
le_dependents = joblib.load('le_dependents.pkl')
le_education = joblib.load('le_education.pkl')
le_self_employed = joblib.load('le_self_employed.pkl')
le_property = joblib.load('le_property.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get data from form
        Gender = request.form['Gender']             # Expecting 'Male' or 'Female'
        Married = request.form['Married']           # Expecting 'Yes' or 'No'
        Dependents = request.form['Dependents']     # Expecting '0', '1', '2', '3+'
        Education = request.form['Education']       # Expecting 'Graduate', 'Not Graduate'
        Self_Employed = request.form['Self_Employed']  # Expecting 'Yes' or 'No'
        ApplicantIncome = float(request.form['ApplicantIncome'])
        CoapplicantIncome = float(request.form['CoapplicantIncome'])
        LoanAmount = float(request.form['LoanAmount'])
        Loan_Amount_Term = float(request.form['Loan_Amount_Term'])
        Credit_History = float(request.form['Credit_History'])
        Property_Area = request.form['Property_Area']  # Expecting 'Urban', 'Semiurban', 'Rural'

        # Encode categorical features using the saved LabelEncoders
        Gender = le_gender.transform([Gender])[0]
        Married = le_married.transform([Married])[0]
        Dependents = le_dependents.transform([Dependents])[0]
        Education = le_education.transform([Education])[0]
        Self_Employed = le_self_employed.transform([Self_Employed])[0]
        Property_Area = le_property.transform([Property_Area])[0]

        # Create feature array
        features = np.array([[Gender, Married, Dependents, Education, Self_Employed,
                              ApplicantIncome, CoapplicantIncome, LoanAmount,
                              Loan_Amount_Term, Credit_History, Property_Area]])

        # Scale features
        features = scaler.transform(features)

        # Predict
        prediction = model.predict(features)[0]

        # Convert prediction to result
        result = 'Approved' if prediction == 1 else 'Rejected'

        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)

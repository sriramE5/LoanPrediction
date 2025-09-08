from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get data from form
        Gender = int(request.form['Gender'])
        Married = int(request.form['Married'])
        Dependents = int(request.form['Dependents'])
        Education = int(request.form['Education'])
        Self_Employed = int(request.form['Self_Employed'])
        ApplicantIncome = float(request.form['ApplicantIncome'])
        CoapplicantIncome = float(request.form['CoapplicantIncome'])
        LoanAmount = float(request.form['LoanAmount'])
        Loan_Amount_Term = float(request.form['Loan_Amount_Term'])
        Credit_History = float(request.form['Credit_History'])
        Property_Area = int(request.form['Property_Area'])

        # Create feature array
        features = np.array([[Gender, Married, Dependents, Education, Self_Employed,
                              ApplicantIncome, CoapplicantIncome, LoanAmount,
                              Loan_Amount_Term, Credit_History, Property_Area]])
        
        features = scaler.transform(features)
        
        prediction = model.predict(features)[0]
        
        result = 'Approved' if prediction == 1 else 'Rejected'
        
        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
# LoanPrediction

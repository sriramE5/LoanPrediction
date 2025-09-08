# Loan Approval Prediction Web Application

A fully functional Loan Approval Prediction web app built using **Flask**, **scikit-learn**, **joblib**, and **NumPy**. The application takes user inputs regarding personal, financial, and property details and predicts whether a loan should be approved or rejected based on a pre-trained machine learning model.

---

## Features

✅ User-friendly web interface with clear input fields  
✅ Input validation using JavaScript  
✅ Model predictions in real-time without page reloads  
✅ Handles categorical features with saved Label Encoders  
✅ Uses `StandardScaler` for proper feature scaling  
✅ Displays results dynamically on the same page  
✅ Mobile-friendly, responsive layout  

---

##  Project Structure

```
LoanPrediction/
├── app.py                   # Main Flask application
├── model.pkl                # Trained machine learning model
├── scaler.pkl               # StandardScaler for feature scaling
├── le_gender.pkl            # Label encoder for Gender
├── le_married.pkl           # Label encoder for Married
├── le_dependents.pkl        # Label encoder for Dependents
├── le_education.pkl         # Label encoder for Education
├── le_self_employed.pkl     # Label encoder for Self Employed
├── le_property_area.pkl     # Label encoder for Property Area
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── templates/
│   └── index.html           # Main HTML template
└── static/
    └── style.css            # CSS styles for the UI
```

---

##  Installation Instructions

### 1. Clone this repository

```bash
git clone https://github.com/your-username/LoanPrediction.git
cd LoanPrediction
```

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Ensure model and encoder files are present

You must have the following files in the project directory:
- `model.pkl`
- `scaler.pkl`
- `le_gender.pkl`
- `le_married.pkl`
- `le_dependents.pkl`
- `le_education.pkl`
- `le_self_employed.pkl`
- `le_property_area.pkl`

If these files are missing, run the training script (`train_model.py`) or use your own prepared files.

### 5. Run the Flask app

```bash
python app.py
```

Visit `http://127.0.0.1:5000/` in your browser to use the app.

---

##  Requirements

The following libraries and versions have been used:

```
Flask==3.1.2
numpy==2.1.0
scikit-learn==1.7.1
joblib==1.5.2
xgboost==3.0.5
```

Install them using:

```bash
pip install -r requirements.txt
```

---

##  Training Model (Optional)

If you want to create the model yourself from a dataset:

1. Prepare your dataset CSV file (e.g., `loan_data.csv`).
2. Encode categorical fields using `LabelEncoder`.
3. Scale numerical features using `StandardScaler`.
4. Train a classification model such as Logistic Regression or Random Forest.
5. Save the model and preprocessors using `joblib.dump()`.

Example training code snippet:

```python
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv('loan_data.csv').dropna()

# Encoding categorical features
le_gender = LabelEncoder().fit(data['Gender'])
data['Gender'] = le_gender.transform(data['Gender'])

# Similarly for other categorical columns...

# Scaling numerical features
scaler = StandardScaler().fit(data[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']])
X_scaled = scaler.transform(data[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']])

# Target variable
y = data['Loan_Status']

# Train the model
model = LogisticRegression()
model.fit(X_scaled, y)

# Save model and preprocessors
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le_gender, 'le_gender.pkl')
# Similarly save other encoders...
```

---

##  Templates and Static Files

- **`templates/index.html`** – Contains the input form and result display using Flask's template engine (Jinja2).
- **`static/style.css`** – Contains CSS styles for the web page layout and responsiveness.

---

##  Usage Instructions

1. Open the web app in your browser.
2. Fill in the fields with valid data:
   - Gender, Married, Dependents, etc.
3. Click the "Predict" button.
4. The prediction ("Approved" or "Rejected") will appear below the form.

---

## ⚙ Notes

✔ The fields must match the values used during model training  
✔ Ensure that inputs such as `Gender`, `Married`, etc., are encoded the same way  
✔ The Flask version used here is `3.1.2` – check compatibility for future updates  
✔ Use a virtual environment to isolate dependencies  

---

##  Future Enhancements

- Deploy using Docker, Heroku, or AWS
- Add user authentication for personalized predictions
- Include data visualization for loan trends
- Improve error handling and validation messages
- Use a database for storing user inputs and predictions

---

## License

This project is open-source and available under the MIT License.

---

## Contact

For issues, questions, or feature requests, feel free to create an issue on GitHub or contact the developer directly.

---

Feel free to copy this file into your `README.md` and modify links, author information, or dataset details as needed!

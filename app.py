import streamlit as st
import joblib
import pandas as pd

# Load the trained model, scaler, and training column names
gb_model = joblib.load('gradient_boosting_model.joblib')
scaler = joblib.load('scaler.joblib')
training_columns = joblib.load('training_columns.joblib')


st.title('Bank Marketing Prediction')
st.write('Predict whether a customer will subscribe to a term deposit.')

st.header('Enter Customer Details:')

# Define the list of features and their types based on the original data
# Hardcoding unique values for categorical features
feature_info = {
    'age': {'type': 'number', 'label': 'Age'},
    'balance': {'type': 'number', 'label': 'Balance'},
    'day': {'type': 'number', 'label': 'Day'},
    'duration': {'type': 'number', 'label': 'Duration (seconds)'},
    'campaign': {'type': 'number', 'label': 'Campaign (number of contacts)'},
    'pdays': {'type': 'number', 'label': 'Pdays (days since last contact)'},
    'previous': {'type': 'number', 'label': 'Previous (number of previous contacts)'},
    'job': {'type': 'category', 'label': 'Job', 'options': ['management', 'technician', 'entrepreneur', 'blue-collar', 'unknown', 'retired', 'admin.', 'services', 'self-employed', 'unemployed', 'housemaid', 'student']},
    'marital': {'type': 'category', 'label': 'Marital Status', 'options': ['married', 'single', 'divorced']},
    'education': {'type': 'category', 'label': 'Education', 'options': ['tertiary', 'secondary', 'unknown', 'primary']},
    'default': {'type': 'category', 'label': 'Credit Default', 'options': ['no', 'yes']},
    'housing': {'type': 'category', 'label': 'Housing Loan', 'options': ['yes', 'no']},
    'loan': {'type': 'category', 'label': 'Personal Loan', 'options': ['no', 'yes']},
    'contact': {'type': 'category', 'label': 'Contact Communication Type', 'options': ['unknown', 'cellular', 'telephone']},
    'month': {'type': 'category', 'label': 'Last Contact Month', 'options': ['may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'jan', 'feb', 'mar', 'apr', 'sep']},
    'poutcome': {'type': 'category', 'label': 'Previous Campaign Outcome', 'options': ['unknown', 'failure', 'other', 'success']}
}

# Create input widgets
input_data = {}
for feature, info in feature_info.items():
    if info['type'] == 'number':
        input_data[feature] = st.number_input(info['label'], value=0, step=1)
    elif info['type'] == 'category':
        input_data[feature] = st.selectbox(info['label'], info['options'])

# Convert input data to a pandas DataFrame
input_df = pd.DataFrame([input_data])

# Preprocess the input data
# Apply one-hot encoding to categorical features
categorical_cols = [col for col, info in feature_info.items() if info['type'] == 'category']
input_df_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

# Ensure the input DataFrame has the same columns as the training data
# Add missing columns and reindex to match the training columns order
missing_cols = set(training_columns) - set(input_df_encoded.columns)
for c in missing_cols:
    input_df_encoded[c] = 0

# Reindex to ensure the order of columns is the same as in the training data
input_df_encoded = input_df_encoded[training_columns]

# Identify numerical columns in the potentially encoded input DataFrame
# These should be the original numerical columns before one-hot encoding
numerical_cols = [col for col, info in feature_info.items() if info['type'] == 'number']

# Scale the numerical features
# Select only the numerical columns from the reindexed DataFrame for scaling
input_df_encoded[numerical_cols] = scaler.transform(input_df_encoded[numerical_cols])


# Make a prediction when a button is clicked
if st.button('Predict'):
    # Use the loaded model to make a prediction
    prediction = gb_model.predict(input_df_encoded)

    # Display the prediction result
    if prediction[0] == 1:
        st.success('Prediction: The customer is likely to subscribe to a term deposit.')
    else:
        st.error('Prediction: The customer is not likely to subscribe to a term deposit.')

st.header('How to Use This App:')
st.write("""
This application uses a trained Gradient Boosting Classifier model to predict whether a bank customer is likely to subscribe to a term deposit based on their attributes and previous interaction history.

To get a prediction, please follow these steps:
1.  **Enter Customer Details:** Fill in the input fields above with the relevant information about the customer.
2.  **Click Predict:** Click the 'Predict' button.
3.  **View Prediction:** The app will display a prediction indicating whether the customer is likely to subscribe ('yes') or not ('no').
""")

st.header('About the Model:')
st.info("""
The model used in this application is a **Gradient Boosting Classifier**.
Gradient Boosting is a powerful machine learning technique that builds a strong predictive model by combining multiple weaker models (typically decision trees) in a sequential manner.
The model was trained on a dataset containing information about bank customers and their responses to previous marketing campaigns.
""")

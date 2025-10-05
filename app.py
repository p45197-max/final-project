import streamlit as st
import joblib
import pandas as pd

# Load the trained model and scaler
gb_model = joblib.load('gradient_boosting_model.joblib')
scaler = joblib.load('scaler.joblib')

st.title('Bank Marketing Prediction')
st.write('Predict whether a customer will subscribe to a term deposit.')

st.header('Enter Customer Details:')

# Define the list of features and their types based on the training data
# Assuming 'X' DataFrame from the notebook contains the feature names and their types
feature_info = {
    'age': {'type': 'number', 'label': 'Age'},
    'balance': {'type': 'number', 'label': 'Balance'},
    'day': {'type': 'number', 'label': 'Day'},
    'duration': {'type': 'number', 'label': 'Duration (seconds)'},
    'campaign': {'type': 'number', 'label': 'Campaign (number of contacts)'},
    'pdays': {'type': 'number', 'label': 'Pdays (days since last contact)'},
    'previous': {'type': 'number', 'label': 'Previous (number of previous contacts)'},
    'job': {'type': 'category', 'label': 'Job', 'options': df['job'].unique().tolist()},
    'marital': {'type': 'category', 'label': 'Marital Status', 'options': df['marital'].unique().tolist()},
    'education': {'type': 'category', 'label': 'Education', 'options': df['education'].unique().tolist()},
    'default': {'type': 'category', 'label': 'Credit Default', 'options': df['default'].unique().tolist()},
    'housing': {'type': 'category', 'label': 'Housing Loan', 'options': df['housing'].unique().tolist()},
    'loan': {'type': 'category', 'label': 'Personal Loan', 'options': df['loan'].unique().tolist()},
    'contact': {'type': 'category', 'label': 'Contact Communication Type', 'options': df['contact'].unique().tolist()},
    'month': {'type': 'category', 'label': 'Last Contact Month', 'options': df['month'].unique().tolist()},
    'poutcome': {'type': 'category', 'label': 'Previous Campaign Outcome', 'options': df['poutcome'].unique().tolist()}
}

# Create input widgets
input_data = {}
for feature, info in feature_info.items():
    if info['type'] == 'number':
        # Use a sensible default value and step for number inputs
        input_data[feature] = st.number_input(info['label'], value=0, step=1)
    elif info['type'] == 'category':
        input_data[feature] = st.selectbox(info['label'], info['options'])

# Convert input data to a pandas DataFrame
input_df = pd.DataFrame([input_data])

# Preprocess the input data
# Apply one-hot encoding to categorical features
categorical_cols = [col for col, info in feature_info.items() if info['type'] == 'category']
input_df_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

# Ensure the input DataFrame has the same columns as the training data (X_train)
# Add missing columns (due to one-hot encoding not covering all categories in input)
missing_cols = set(X_train.columns) - set(input_df_encoded.columns)
for c in missing_cols:
    input_df_encoded[c] = 0

# Ensure the order of columns is the same as in the training data
input_df_encoded = input_df_encoded[X_train.columns]

# Identify numerical columns in the potentially encoded input DataFrame
# We need to consider that original numerical columns remain, while categorical are now dummy variables (integers/booleans)
numerical_cols_after_encoding = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
# Filter for columns that are actually in the input_df_encoded after alignment
numerical_cols_in_input = [col for col in numerical_cols_after_encoding if col in input_df_encoded.columns]


# Scale the numerical features
input_df_encoded[numerical_cols_in_input] = scaler.transform(input_df_encoded[numerical_cols_in_input])

# Display the preprocessed input data (optional, for debugging)
# st.write("Preprocessed Input Data:")
# st.write(input_df_encoded)

# Make a prediction when a button is clicked
if st.button('Predict'):
    # Use the loaded model to make a prediction
    prediction = gb_model.predict(input_df_encoded)

    # Display the prediction result
    if prediction[0] == 1:
        st.success('Prediction: The customer is likely to subscribe to a term deposit.')
    else:
        st.error('Prediction: The customer is not likely to subscribe to a term deposit.')
st.header('Enter Customer Details:')

# Define the list of features and their types based on the training data
# Assuming 'X' DataFrame from the notebook contains the feature names and their types
# Hardcoding unique values for categorical features as df is not available in app.py
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
        # Use a sensible default value and step for number inputs
        input_data[feature] = st.number_input(info['label'], value=0, step=1)
    elif info['type'] == 'category':
        input_data[feature] = st.selectbox(info['label'], info['options'])

# Convert input data to a pandas DataFrame
input_df = pd.DataFrame([input_data])

# Load the trained model and scaler
gb_model = joblib.load('gradient_boosting_model.joblib')
scaler = joblib.load('scaler.joblib')

st.title('Bank Marketing Prediction')
st.write('Predict whether a customer will subscribe to a term deposit.')

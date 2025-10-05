import streamlit as st
import joblib
import pandas as pd
import pickle # Import pickle

# Load the trained model, scaler, and training column names
# Make sure these files (gradient_boosting_model.pkl, scaler.pkl, training_columns.joblib)
# are in the same directory as your app.py file when deploying.
try:
    # Load model and scaler using pickle
    with open('gradient_boosting_model.pkl', 'rb') as f:
        gb_model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Load training columns using joblib
    training_columns = joblib.load('training_columns.joblib')

except FileNotFoundError:
    st.error("Model, scaler, or training columns file not found. Please ensure they are in the correct directory.")
    st.stop() # Stop the app if files are not found


st.title('Bank Marketing Prediction')
st.write('Predict whether a customer will subscribe to a term deposit.')

st.header('Enter Customer Details:')

# Define the list of features and their types based on the original data
# Hardcoding unique values for categorical features
feature_info = {
    'age': {'type': 'number', 'label': 'Age', 'min_value': 0, 'max_value': 120, 'step': 1},
    'balance': {'type': 'number', 'label': 'Balance', 'min_value': -10000, 'max_value': 100000, 'step': 1}, # Example ranges
    'day': {'type': 'number', 'label': 'Day', 'min_value': 1, 'max_value': 31, 'step': 1},
    'duration': {'type': 'number', 'label': 'Duration (seconds)', 'min_value': 0, 'max_value': 5000, 'step': 1}, # Example ranges
    'campaign': {'type': 'number', 'label': 'Campaign (number of contacts)', 'min_value': 1, 'max_value': 60, 'step': 1}, # Example ranges
    'pdays': {'type': 'number', 'label': 'Pdays (days since last contact)', 'min_value': -1, 'max_value': 900, 'step': 1}, # Example ranges, -1 for never contacted
    'previous': {'type': 'number', 'label': 'Previous (number of previous contacts)', 'min_value': 0, 'max_value': 300, 'step': 1}, # Example ranges
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
        # Use min_value, max_value, and step from feature_info
        input_data[feature] = st.number_input(info['label'],
                                              min_value=info.get('min_value', 0),
                                              max_value=info.get('max_value', None),
                                              step=info.get('step', 1),
                                              value=info.get('min_value', 0)) # Set default value
    elif info['type'] == 'category':
        input_data[feature] = st.selectbox(info['label'], info['options'])

# Convert input data to a pandas DataFrame
input_df = pd.DataFrame([input_data])

# Preprocess the input data
# Apply one-hot encoding to categorical features
categorical_cols = [col for col, info in feature_info.items() if info['type'] == 'category']
input_df_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

# Ensure the input DataFrame has the same columns as the training data
# Reindex to ensure the order of columns is the same as in the training data
# This will also add any missing columns that were present in training but not in the input
input_df_encoded = input_df_encoded.reindex(columns=training_columns, fill_value=0)

# Identify numerical columns based on feature_info
numerical_cols = [col for col, info in feature_info.items() if info['type'] == 'number']


# Scale the numerical features
# Select only the numerical columns from the reindexed DataFrame for scaling
numerical_cols_in_encoded_df = [col for col in numerical_cols if col in input_df_encoded.columns]
input_df_encoded[numerical_cols_in_encoded_df] = scaler.transform(input_df_encoded[numerical_cols_in_encoded_df])


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

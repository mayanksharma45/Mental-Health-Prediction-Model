import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import sys
from src.logger import logging
from src.exception import CustomException

# Load the dataset
def load_dataset():
    try:
        data = pd.read_csv(r"src\data\survey.csv")
        return data
    except Exception as e:
        raise CustomException(e, sys)

# Data cleaning function
def clean_dataset(data):
    try:
        logging.info("Starting data cleaning...")

        # Drop unnecessary columns
        data.drop(['comments', 'state', 'Timestamp'], axis=1, inplace=True)

        # Handling missing values
        defaultInt = 0
        defaultString = 'NaN'

        numerical_features = ['Age']
        categorical_features = ['Gender', 'Country', 'self_employed', 'family_history', 'treatment', 'work_interfere',
                                'no_employees', 'remote_work', 'tech_company', 'anonymity', 'leave', 'mental_health_consequence',
                                'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview', 
                                'phys_health_interview', 'mental_vs_physical', 'obs_consequence', 'benefits', 'care_options', 
                                'wellness_program', 'seek_help']

        for feature in data.columns:
            if feature in numerical_features:
                data[feature].fillna(defaultInt, inplace=True)
            elif feature in categorical_features:
                data[feature].fillna(defaultString, inplace=True)
            else:
                logging.warning(f"Unexpected feature {feature} found.")

        logging.info("Handling missing values completed.")

        # Standardizing Gender values
        male_labels = {"male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man", "msle", "mail", "malr", "cis man", "cis male"}
        female_labels = {"cis female", "f", "female", "woman", "femake", "female ", "cis-female/femme", "female (cis)", "femail"}
        trans_labels = {"trans-female", "non-binary", "queer", "genderqueer", "androgyne", "agender", "trans woman", "neuter", "female (trans)"}

        def standardize_gender(value):
            value = value.lower().strip()
            if value in male_labels:
                return "male"
            elif value in female_labels:
                return "female"
            elif value in trans_labels:
                return "trans"
            else:
                return "other"  # Assign 'other' to unexpected values

        data['Gender'] = data['Gender'].apply(standardize_gender)

        # Remove invalid gender entries
        data = data[~data['Gender'].isin(["A little about you", "p"])]

        logging.info("Gender values standardized.")

        # Handle outliers in Age
        # checking quantiles
        q1 = data['Age'].quantile(0.25)   # first quartile, q1
        q3 = data['Age'].quantile(0.75)   # third quartile, q3

        #calculating iqr
        iqr = q3 - q1   # inter-quartile range, iqr

        # Calculation upper_limit and lower_limit
        upper_limit = q3 + 1.5*iqr   
        lower_limit = q1 - 1.5*iqr
        upper_limit, lower_limit

        # imputing outliers by defing a function
        def limit_imputer(value):
            if value > upper_limit:
                return upper_limit
            if value < lower_limit:
                return lower_limit
            else:
                return value

        logging.info("Outlier treatment completed.")

        # Replace missing self-employed values
        data['self_employed'].replace([defaultString], 'No', inplace=True)

        # Drop 'Country' since it's not a useful feature
        data.drop(columns=['Country', "leave"], inplace=True)

        return data
    except Exception as e:
        raise CustomException(e, sys)

# Encoding categorical variables
def encode_features(data, target_column):
    try:
        categorical_features = ['Gender', 'self_employed', 'family_history', 'work_interfere',
                                'no_employees', 'remote_work', 'tech_company', 'anonymity', 
                                'mental_health_consequence', 'phys_health_consequence', 'coworkers', 
                                'supervisor', 'mental_health_interview', 'phys_health_interview', 
                                'mental_vs_physical', 'obs_consequence', 'benefits', 'care_options', 
                                'wellness_program', 'seek_help']

        label_encoders = {}

        # Loop over each categorical feature
        for col in categorical_features:
            le = LabelEncoder()

            # Combine train and test labels for the feature
            all_labels = pd.concat([data[col], data[col]])  # Combine both train and test data
            le.fit(all_labels)  # Fit the encoder on the combined labels

            # Apply transformation on the column
            data[col] = le.transform(data[col])

            # Save the encoder for future use
            label_encoders[col] = le

        # Explicit encoding for the target variable ('Yes' -> 1, 'No' -> 0)
        data[target_column] = data[target_column].map({'Yes': 1, 'No': 0})

        return data, label_encoders
    except Exception as e:
        raise CustomException(e, sys)
    
logging.info(f"Encoded applied successfully")

# Split dataset into training and testing sets
def preprocess_data(data, target_column):
    try:
        X = data.drop(columns=[target_column])  # Features
        y = data[target_column]  # Target variable

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        raise CustomException(e, sys)
    
logging.info(f"Preprocessing applied successfully")

logging.info(f"Scaling the X_train and X_test")

# Feature scaling
def scale_features(X_train, X_test):
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Save the scaler for future use
        with open("artifacts/scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        return X_train_scaled, X_test_scaled
    except Exception as e:
        raise CustomException(e, sys)

# Main function to prepare dataset
def prepare_data(target_column):
    try:
        data = load_dataset()
        data = clean_dataset(data)
        data, label_encoders = encode_features(data, target_column)
        X_train, X_test, y_train, y_test = preprocess_data(data, target_column)
        X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

        # Save label encoders
        with open("artifacts/label_encoders.pkl", "wb") as f:
            pickle.dump(label_encoders, f)

        return X_train_scaled, X_test_scaled, y_train, y_test
    
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    target_column = "treatment"  # Define target variable
    prepare_data(target_column)
    print(" Dataset preparation completed successfully!")

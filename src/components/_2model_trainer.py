import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from scipy.sparse import csr_matrix
from sklearn.metrics import classification_report, accuracy_score
from src.components._1data_preparation import prepare_data
from src.logger import logging
from src.utils import evaluate_models
from src.exception import CustomException

import pandas as pd
import numpy as np
import sys

# Train the model
def choose_best_model(X_train, X_test, y_train, y_test):
    try:
        if not isinstance(y_train, (np.ndarray, pd.Series)):
            raise ValueError("The `y_train` parameter must be a pandas Series or numpy array.")
        
        unique_classes = len(np.unique(y_train)) if isinstance(y_train, np.ndarray) else y_train.nunique()

        models = {
                    "Logistic Regression": LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=500),
                    "Random Forest Classifier": RandomForestClassifier(n_estimators=100),
                    "XGB Classifier": XGBClassifier(objective="multi:softmax", num_class=unique_classes), 
                    "CatBoosting Classifier": CatBoostClassifier(verbose=False),
                    "AdaBoost Classifier": AdaBoostClassifier(),
                    "Gradient Boosting Classifier": GradientBoostingClassifier(),
                    "Support Vector Classifier": SVC(probability=True)  # Ensure probability=True for AUC-ROC
                }

        model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)
        
        # Get best model score from dict
        best_model_score = max(sorted(model_report.values()))

        # Get best model name from dict
        best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
        best_model = models[best_model_name]

        if best_model_score < 0.6:
            raise CustomException("No best model found with sufficient accuracy.")
        
        logging.info(f"Best found model on both training and testing dataset.")

        predicted = best_model.predict(X_test)
        Accuracy_Score = accuracy_score(y_test, predicted)
        print(f"Accuracy_Score: {Accuracy_Score}")
        
        # Save the model
        with open("artifacts/model.pkl", "wb") as f:
            pickle.dump(best_model, f)
        
        return best_model, Accuracy_Score

    except ValueError as e:
        logging.error(f"ValueError: {e}")
        raise CustomException(f"ValueError: {e}", sys)
    except Exception as e:
        raise CustomException(e, sys)

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test) 
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    except Exception as e:
        raise CustomException(e, sys)

# Main function to train and evaluate the model
if __name__ == "__main__":
    try:
        target_column = "treatment"  # Replace with your target variable
        X_train, X_test, y_train, y_test = prepare_data(target_column)
        model, accuracy = choose_best_model(X_train, X_test, y_train, y_test)
        print("Model training completed!")
        evaluate_model(model, X_test, y_test)
    except Exception as e:
        raise CustomException(e, sys)

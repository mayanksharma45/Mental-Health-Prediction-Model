import pickle
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from src.components._1data_preparation import prepare_data
from src.logger import logging
from src.utils import evaluate_models
from src.exception import CustomException

import pandas as pd
import numpy as np
import sys
import lime
import lime.lime_tabular
import warnings
warnings.filterwarnings("ignore")

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
            "Support Vector Classifier": SVC(probability=True)
        }

        model_report = evaluate_models(X_train, y_train, X_test, y_test, models)
        best_model_score = max(sorted(model_report.values()))
        best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
        best_model = models[best_model_name]

        if best_model_score < 0.6:
            raise CustomException("No best model found with sufficient accuracy.")

        logging.info(f"Best model found: {best_model_name}")
        
        best_model.fit(X_train, y_train)
        predicted = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, predicted)
        print(f"Accuracy_Score: {accuracy}")
        
        with open("artifacts/model.pkl", "wb") as f:
            pickle.dump(best_model, f)
        
        return best_model, accuracy

    except ValueError as e:
        logging.error(f"ValueError: {e}")
        raise CustomException(f"ValueError: {e}", sys)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    except Exception as e:
        raise CustomException(e, sys)


import shap
import logging
import sys
from sklearn.svm import SVC  # Ensure you import SVC for the special case

def explain_model_lime(model, X_train, y_train):
    try:
        # Ensure the model is explicitly set to AdaBoostClassifier
        model.fit(X_train, y_train)

        # Create LIME explainer for tabular data
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train,  # Directly pass X_train (it can be ndarray or pandas DataFrame)
            mode='classification',  # Specify classification problem
            feature_names=None,  # Feature names can be None for ndarray
            class_names=np.unique(y_train),  # Class labels (ensure it is a numpy array)
            discretize_continuous=True  # Discretize continuous features for better explanation
        )

        # Choose an instance to explain (select instance index)
        instance_idx = 0
        instance = X_train[instance_idx]  # For ndarray, use index directly

        # Generate LIME explanation for the selected instance
        explanation = explainer.explain_instance(
            instance,  # Features of the selected instance
            model.predict_proba,  # The model's probability prediction function
            num_features=10  # Show top 10 features contributing to the prediction
        )

        # Plot the explanation
        explanation.as_pyplot_figure()
        plt.show()

    except Exception as e:
        print(f"Error: {e}")
        raise CustomException(e, sys)





if __name__ == "__main__":
    try:
        target_column = "treatment"
        X_train, X_test, y_train, y_test = prepare_data(target_column)
        model, accuracy = choose_best_model(X_train, X_test, y_train, y_test)
        print("Model training completed!")
        evaluate_model(model, X_test, y_test)
        explain_model_lime(model, X_train, y_train)
    except Exception as e:
        raise CustomException(e, sys)

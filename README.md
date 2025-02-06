# Mental Health Prediction System

This project uses machine learning to predict mental health conditions based on various factors.  It includes data preprocessing, model training, and a Streamlit-based inference application.

## 1. Dataset Preprocessing Steps

The following steps were taken to preprocess the dataset before model training:

1. **Data Cleaning:**
    * Handled missing values using imputation with mean/median/mode and removal of rows/columns with excessive missing data techniques.
    * Removed duplicate rows.
    * Handled outliers in numerical features using IQR method.
    * Converted data types to numerical formats.

2. **Feature Engineering:**
    * Transformed or scaled numerical features using standardization to ensure they have similar ranges.  *Specify the scaling method used.*

3. **Encoding Categorical Variables:**
    * Converted categorical features into numerical representations using label encoding and mapping method.

4. **Data Splitting:**
    * Divided the dataset into training and test sets in split ratio 80/20.

## 2. Model Selection Rationale

*Describe the model(s) you considered and why you chose the final model.*  For example:

    * **Model Candidates:**  We explored several models, including Logistic Regression, Random Forest, Gradient Boosting, and Support Vector Machines.
    * **Rationale:** We selected AdaBoost Classifier because it provided the best performance in terms of accuracy, precision, recall, F1-score on the test set. 
    It also offered a good balance between performance and interpretability.

## 3. How to Run the Inference Script (Streamlit App)

These steps will guide you through running the Streamlit app for inference:

1. **Clone the Repository:**

   ```bash
   git clone [https://github.com/mayanksharma45/Mental-Health-Prediction-Model.git](https://www.google.com/search?q=https://github.com/mayanksharma45/Mental-Health-Prediction-Model.git)
   cd Mental-Health-Prediction-Model

2. **Create a Virtual Environment**
conda activate -n health python==3.12  # Create a virtual environment
conda activate health  # Activate on Windows

3. **Install Dependencies**
pip install -r requirements.txt

4. **Prepare Model Artifacts**
Ensure that your trained model (model.pkl), scaler (scaler.pkl), and label encoders (label_encoders.pkl) are located in the artifacts/ directory.

5. **Run the Streamlit App**
streamlit run model_testing.py

6. **Access the App**
Streamlit will provide a URL in the terminal (usually http://localhost:8501). Open this URL in your web browser.

## 4. UI Usage Instructions

#### Streamlit Web UI

- __Input Features__: The Streamlit app will display input fields for various features. Enter the appropriate values for each feature. Provide a brief description of each feature and its expected input format (e.g., numerical, categorical).
__Prediction__: Click the "Predict" button to generate the prediction.
__Output__: The predicted mental health condition (e.g., "Needs Treatment," "No Treatment Needed") will be displayed on the screen.
__Error Handling__: If there are any issues with the input or during prediction, appropriate error messages will be displayed.


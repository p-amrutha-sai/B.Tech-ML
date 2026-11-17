# Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to leave a service provider. The project uses customer demographics, account information, subscribed services, and billing details to identify customers who may churn.

The project includes data exploration, preprocessing, model training, evaluation, and an interactive web application built using Flask.

## 1. Problem Statement

Customer churn occurs when customers stop using a company's services. Predicting churn helps businesses identify customers who are at risk of leaving and take preventive actions to improve customer retention.

The objective of this project is to build a machine learning model that predicts whether a telecom customer will churn based on historical customer data.

## 2. Dataset

**Dataset:** IBM Telco Customer Churn Dataset

The dataset contains information about 7,043 telecom customers and 21 columns.

Some important features include:

- `gender`: Customer gender
- `SeniorCitizen`: Whether the customer is a senior citizen
- `Partner`: Whether the customer has a partner
- `Dependents`: Whether the customer has dependents
- `tenure`: Number of months the customer has stayed with the company
- `InternetService`: Type of internet service
- `Contract`: Type of customer contract
- `MonthlyCharges`: Monthly amount charged
- `TotalCharges`: Total amount charged
- `Churn`: Whether the customer left the company

The target variable is `Churn`:
- `Yes` – Customer left the company.
- `No` – Customer stayed with the company.

## 3. Project Structure

```text
customer-churn-prediction/
│
├── dataset/
│   └── Telco-Customer-Churn.csv
│
├── eda/
│   └── eda.py
│
├── src/
│   └── train.py
│
├── model/
│   └── churn_model.pkl
│
├── web/
│   └── index.html
│   └── styles.css
│
├── app.py
├── requirements.txt
└── README.md
```

### Folder and File Description

| File / Folder | Purpose |
|---|---|
| `dataset/` | Contains the customer churn dataset |
| `eda/eda.py` | Performs exploratory data analysis |
| `src/train.py` | Preprocesses data, trains and evaluates models |
| `model/` | Stores the trained machine learning model |
| `app.py` | Runs the interactive application |
| `requirements.txt` | Lists the required Python libraries |
| `README.md` | Project documentation |

## 4. Technologies Used

- **Python** – Main programming language
- **Pandas** – Data loading and manipulation
- **NumPy** – Numerical operations
- **Matplotlib & Seaborn** – Data visualization
- **Scikit-learn** – Data preprocessing and machine learning
- **Joblib** – Saving and loading the trained model
- **Flask** – Interactive web application

## 5. Project Workflow

### Step 1: Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand the dataset and identify patterns.

The following operations were performed:

- Examined the first few records and dataset dimensions.
- Checked data types and missing values.
- Identified duplicate records.
- Analyzed the distribution of the target variable.
- Visualized churn distribution.
- Studied churn patterns across contract types.
- Examined the relationship between customer tenure and churn.

**Dataset observations:**

- Total records: 7,043
- Total features: 21
- Customers who stayed: 5,174
- Customers who churned: 1,869
- Churn rate: Approximately 26.54%

The `TotalCharges` column initially contained blank values stored as strings. These were converted to numeric values. The resulting 11 missing values corresponded to customers with zero tenure, so they were filled with zero.

### Step 2: Data Preprocessing

Before training, the dataset was prepared for machine learning.

The following preprocessing steps were applied:

1. Removed `customerID` because it is an identifier and does not provide useful predictive information.
2. Converted `TotalCharges` into numeric format.
3. Replaced missing `TotalCharges` values with zero.
4. Converted the target variable into numerical values:
   - No = 0
   - Yes = 1
5. Separated numerical and categorical features.

**Numerical features:**
- SeniorCitizen
- tenure
- MonthlyCharges
- TotalCharges

**Categorical features:**

The remaining customer and service-related features.

**Feature transformation:**

- StandardScaler was used to scale numerical features.
- OneHotEncoder was used to convert categorical features into numerical representations.

A `ColumnTransformer` was used to apply these transformations to the appropriate columns.

### Step 3: Model Training

Three classification algorithms were trained and compared:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

The dataset was divided into training and testing sets:

- Training data: 80%
- Testing data: 20%

`stratify=y` was used to preserve the class distribution in both sets.

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Class weighting was used to help address the imbalance between churn and non-churn customers.

### Step 4: Model Evaluation

The following results were obtained on the test dataset:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 73.81% | 50.43% | 78.34% | 61.36% |
| Decision Tree | 73.60% | 50.27% | 49.47% | 49.87% |
| Random Forest | 77.00% | 55.87% | 63.64% | 59.50% |

**Selected Model: Logistic Regression**

Logistic Regression was selected because it achieved the highest F1-score of approximately 61.36%.

Although Random Forest achieved higher accuracy, Logistic Regression provided a better balance between precision and recall for identifying customers likely to churn.

Recall is particularly important in this project because missing a customer who is actually going to churn can mean losing an opportunity to retain them.

### Step 5: Model Saving

The best-performing model was saved using Joblib:

```python
joblib.dump(best_model, "model/churn_model.pkl")
```

The saved file contains the fitted machine learning pipeline, including the preprocessing transformations and trained classifier.

This allows the application to make predictions without retraining the model every time it starts.

### Step 6: Streamlit Application

An interactive web application was developed using Streamlit.

Users can enter customer details such as:

- Customer tenure
- Contract type
- Internet service
- Monthly charges
- Total charges
- Payment method
- Other subscribed services

The application uses the saved model to predict whether the customer is likely to churn.

It also displays the predicted churn probability.

## 6. Installation and Setup

### Prerequisites

- Python 3
- pip

### Create a Virtual Environment

```bash
python3 -m venv venv
```

Activate it on macOS/Linux:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## 7. How to Run the Project

Run the following commands from the project root directory.

### Run Exploratory Data Analysis

```bash
python eda/eda.py
```

### Train the Machine Learning Models

```bash
python src/train.py
```

This trains the three classifiers, evaluates their performance, selects the model with the highest F1-score, and saves it inside the `model/` directory.

### Launch the Application
In the same virtual environment
```bash
python app.py
```

Open the local URL displayed in the terminal, usually:

```text
http://127.0.0.1:5000/
```

Enter customer details and click the prediction button to see the churn prediction.

## 8. Future Improvements

Some possible improvements to the project include:

- Hyperparameter tuning to improve model performance.
- Cross-validation for more reliable model evaluation.
- Feature importance analysis to identify the strongest churn indicators.
- Experimenting with additional algorithms such as XGBoost.
- Threshold tuning to improve churn detection.
- Improving the Streamlit interface with more visualizations.
- Adding customer retention recommendations based on prediction results.
- Deploying the application to a cloud platform.

## 9. Key Learning Outcomes

Through this project, the following concepts were explored:

- Exploratory Data Analysis
- Data cleaning and preprocessing
- Handling missing values
- Encoding categorical variables
- Feature scaling
- Classification algorithms
- Handling imbalanced datasets
- Model evaluation using precision, recall, and F1-score
- Building machine learning pipelines
- Saving and loading trained models
- Building interactive ML applications using Streamlit

## 10. Disclaimer

This is an educational machine learning project built using a publicly available telecom customer churn dataset. The model provides predictions based on historical data and should not be treated as a production-ready customer retention system without further validation and testing.

---

**Project:** Customer Churn Prediction  
**Domain:** Machine Learning / Classification  
**Language:** Python  
**Interface:** Flask

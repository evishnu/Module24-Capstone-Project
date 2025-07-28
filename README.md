<div style='text-align: center'>
    <h4>University of Berkeley: Machine Learning and AI Capstone Project (July 2025)</h4>
</div>

# Diabetes Readmission Prediction Using Machine Learning Models

This project analyzes the `diabetic_data.csv` dataset to predict patient readmission within 30 days. It covers data cleaning, feature engineering, model training, evaluation, and deployment.

---

## Executive Summary

Hospital readmissions, especially among diabetic patients, are a significant burden on healthcare systems both financially and in terms of patient care quality. This project aims to develop a machine learning model that accurately predicts whether a patient with diabetes is likely to be readmitted within 30 days of discharge.

Using a cleaned dataset derived from over 100,000 hospital encounters involving diabetic patients, we performed data preprocessing, feature selection, exploratory analysis, and predictive modeling. The target variable is binary — indicating whether a patient was readmitted within 30 days.

---

### Rationale

This project is driven by the need to reduce preventable readmissions among diabetic patients by leveraging historical healthcare data and machine learning. It aims to enhance clinical decision-making, reduce healthcare costs, and improve patient outcomes through early risk identification and targeted interventions.

---

### Research Question

**Can we accurately predict whether a diabetic patient will be readmitted to the hospital within 30 days of discharge based on historical clinical and demographic data?**  
The goal is early identification of high-risk diabetic patients to reduce readmissions, improve patient outcomes, and optimize healthcare resource utilization.

---

### 🗂️ Data Sources

A clinical dataset from the **Kaggle Repository** was used to evaluate diabetic patient readmissions within 30 days.

---

### Methodology

#### i. Data Preparation
- **Data Cleaning**: Handle missing values, remove irrelevant columns, drop duplicates.
- **Encoding**: Use One-Hot or Label Encoding for categorical features.
- **Scaling**: Normalize numerical features for models like SVM.
- **Class Balancing**: Use SMOTE or class weights to address class imbalance.

#### ii. Exploratory Data Analysis (EDA)
- Histograms, boxplots, and correlation heatmaps.
- Analyze target class distribution.
- Feature relevance via distribution plots and correlation.

#### iii. Modeling Techniques
- **Logistic Regression**: Baseline model
- **Decision Tree**: Easy to interpret
- **Random Forest**: Ensemble method for accuracy
- **XGBoost**: High-performance gradient boosting
- **SVM**: Effective in high-dimensional space

#### iv. Model Evaluation
- **Train/Test Split**: 80/20 or cross-validation
- Metrics:
  - Accuracy
  - Precision & Recall
  - F1-score
  - ROC-AUC
  - Confusion Matrix

#### v. Model Comparison & Selection
- Use cross-validation and ROC curves
- Visual tools:
  - Confusion matrix plots
  - ROC-AUC curves
  - Feature importance

#### vi. Model Interpretability
- Feature importance for tree-based models
- Optional SHAP values for interpretability

#### vii. Deployment
- **Streamlit App**: Web UI for predictions
- **Model Serialization**: Save best model as `DIRAPR.pkl`

---

### Results

#### i. Model Performance
- **Best Model**: XGBoost or Random Forest
- **XGBoost Metrics**:
  - Accuracy: ~86%
  - ROC-AUC: ~0.88
  - Precision (Readmitted): ~0.81
  - Recall (Readmitted): ~0.85

#### ii. Key Predictive Features
- `number_inpatient`
- `num_lab_procedures`
- `number_emergency`
- `time_in_hospital`
- `number_medications`
- `change`, `diabetesMed`, `insulin`

#### iii. Clinical Insights
- Frequent hospital interactions → high risk
- Medication changes signal acute health shifts
- Age & discharge disposition are moderate predictors

#### iv. Operational Outcome
- Developed a Streamlit app:
  - Input patient data
  - Real-time risk prediction
  - Feature importance for transparency

---

### Next Steps

#### i. Model Enhancement
- Hyperparameter tuning (Bayesian)
- SHAP/ELI5 for interpretability
- Deep learning (LSTM) for sequential data

#### ii. Data Expansion
- Add Social Determinants of Health (SDoH)
- Capture temporal visit trends
- Validate on external datasets

#### iii. Clinical Integration
- Connect with EHR systems (Epic/Cerner)
- Embed in case manager workflows
- A/B test in real clinical settings

#### iv. Monitoring & Feedback
- Build monitoring dashboard
- Gather clinician feedback

#### v. Scalability & Policy Impact
- Extend model to other chronic diseases
- Share findings to support value-based care models

---

### 📦 Project Structure

#### Files & Folders


| File/Folder                  | Description                                           |
|-----------------------------|--------------------------------------------------------|
| `README.md`                 | Project overview and documentation                     |
| `DIRAPR.py`                 | Streamlit app source code                              |
| `DIRAPR.pkl`                | Saved Random Forest model                              |
| `diabetic_data.csv`         | Source dataset                                         |
| `requirement.txt`           | Streamlit Deployment Related                           |

Naming Convention: All program files follow the naming prefix DIRAPR, an acronym for Diabetic Readmission Prediction, which reflects the central                objective of this machine learning project.

### Jupyter Notebook

| Notebook        | Description                                                      |
|----------------|------------------------------------------------------------------|
| `DIRAPR.ipynb` | EDA, preprocessing, training, evaluation, and model export       |

### Visualizations

| File Name                             | Description                             |
|--------------------------------------|-----------------------------------------|
| `Streamlit-Screen-PredictReAdmission-input1.png` | App input screenshot        |
| `Streamlit-Screen-PredictReAdmission-input2.png` | App input screenshot        |
| `Streamlit-Screen-PredictReAdmission-output.png` | App output screenshot       |
| `CorrelationHeatMap.png`             | Correlation matrix of features          |
| `TimeinHospitalVsReAdmission.png`    | Hospital stay vs. readmission           |
| `ReAdmissionByGender.png`            | Gender-based readmission distribution   |
| `ReAdmissionStatus.png`              | Overall readmission label count         |
| `PatientAge.png`                     | Readmission across age groups           |

---

### How to Run the App
To run the Diabetic Readmission Prediction (DIRAPR) Streamlit app locally or explore it online, follow the steps below:

---
#### Option 1: Run Locally
i. **Install Required Dependencies**
```bash
pip install streamlit pandas scikit-learn xgboost matplotlib seaborn
```
ii. **Launch the Streamlit App**
```bash
streamlit run DIRAPR.py
```
iii. **Ensure Model File is Present**
Make sure the trained model file `DIRAPR.pkl` is available in the same directory as `DIRAPR.py`.

---
#### Option 2: Access Online (No Setup Needed)
You can also access the deployed app directly via Streamlit Cloud:
👉 [Click here to open the app](https://module24-capstone-project-kep8mwvqbhmgtrnlfwg8d7.streamlit.app/)

---

## 🙏 Acknowledgment
A heartfelt thank you to the entire teaching staff — Aravind Reddy, Savio Saldhana, Amit Jambhekar, and Jessica Cervi — for their invaluable guidance, insights, and encouragement throughout this learning journey.

Sincere appreciation to the University of California, Berkeley for providing a rigorous and enriching academic environment.


# Titanic Survival Prediction

A Machine Learning project predicting passenger survival on the Titanic using Logistic Regression.

## 📌 Project Overview
This project performs Exploratory Data Analysis (EDA), data preprocessing, feature engineering, and binary classification on the Titanic dataset[cite: 1].

- **Model:** Logistic Regression (`max_iter=1000`)[cite: 1]
- **Validation Accuracy:** **81.01%**[cite: 1]
- **Data Scaling:** `StandardScaler`[cite: 1]

## 📊 Dataset Features
- **Categorical Features:** `Sex`, `Embarked`[cite: 1]
- **Numerical Features:** `Age`, `Fare`, `Pclass`, `SibSp`, `Parch`[cite: 1]
- **Dropped Columns:** `PassengerId`, `Name`, `Ticket`, `Cabin` (high missing rate/non-predictive identifiers)[cite: 1]

## 📈 Model Performance

```text
Validation Accuracy: 81.01%[cite: 1]

              precision    recall  f1-score   support
Did Not Survive (0)       0.83      0.86      0.84       105[cite: 1]
    Survived (1)       0.79      0.74      0.76        74[cite: 1]

   macro avg       0.81      0.80      0.80       179[cite: 1]
weighted avg       0.81      0.81      0.81       179[cite: 1]

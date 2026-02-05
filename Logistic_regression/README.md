# Logistic Regression – Heart Attack Possibility Prediction

This README is written as **complete revision notes** for my Logistic Regression notebook.
If I read this file again in the future, I should be able to **recall the full problem, the theory behind Logistic Regression, and every step I performed in the notebook**, along with the reasoning behind each step.

This README is meant for **conceptual revision + clear understanding**, not just project documentation.

---

## 📌 Problem Overview

The aim of this project is to **predict the possibility of a heart attack** for a person based on medical and lifestyle parameters using **Logistic Regression**.

This is a **classification problem**, not regression, because:

* The output is **categorical** (heart attack risk: yes / no)
* We are predicting **probability and class**, not a continuous value

---

## 📊 Dataset Information

* **Dataset Name:** Health Care Dataset on Heart Attack Possibility
* **Source:** Kaggle
* **Link:** [https://www.kaggle.com/datasets/nareshbhat/health-care-data-set-on-heart-attack-possibility](https://www.kaggle.com/datasets/nareshbhat/health-care-data-set-on-heart-attack-possibility)

### Dataset Description

Each row represents one patient and contains medical measurements and health-related attributes that influence heart health.

### Common Features in the Dataset

| Feature  | Meaning                                               |
| -------- | ----------------------------------------------------- |
| age      | Age of the patient                                    |
| sex      | Gender of the patient                                 |
| cp       | Chest pain type                                       |
| trtbps   | Resting blood pressure                                |
| chol     | Cholesterol level                                     |
| fbs      | Fasting blood sugar                                   |
| restecg  | Resting ECG results                                   |
| thalachh | Maximum heart rate achieved                           |
| exng     | Exercise induced angina                               |
| oldpeak  | ST depression induced by exercise                     |
| slp      | Slope of ST segment                                   |
| caa      | Number of major vessels                               |
| thall    | Thalassemia                                           |
| output   | **Target variable** (1 = higher risk, 0 = lower risk) |

---

## 🎯 Objective of Using Logistic Regression

The objective was to:

* Understand **binary classification**
* Learn how Logistic Regression predicts **probabilities**
* Classify patients into **risk categories** (heart attack possible or not)

---

## 🧠 Theory: Concepts Used in This Notebook

### 1️⃣ Why Not Linear Regression?

Linear Regression:

* Predicts continuous values
* Can produce outputs outside the range [0, 1]

For medical risk prediction, we need:

* Output as **probability**
* Clear class decision

Hence, Linear Regression is not suitable.

---

### 2️⃣ Logistic Regression (Core Idea)

Logistic Regression models the probability of an event using the **sigmoid function**:

```
P(y=1) = 1 / (1 + e^-(w·x + b))
```

Where:

* Output is always between **0 and 1**
* Values close to 1 → higher risk
* Values close to 0 → lower risk

---

### 3️⃣ Decision Boundary

* A threshold (usually 0.5) is applied
* If probability ≥ 0.5 → class 1 (risk present)
* Else → class 0 (risk not present)

---

### 4️⃣ Cost Function (Log Loss)

Logistic Regression minimizes **log loss**, not MSE:

* Penalizes wrong confident predictions heavily
* Suitable for classification problems

---

## 🔧 Step-by-Step: What I Did and Why

### 1️⃣ Importing Required Libraries

Used:

* `pandas` for data handling
* `numpy` for numerical operations
* `matplotlib / seaborn` for visualization
* `sklearn` for model training and evaluation

Why:
These libraries form the standard ML classification pipeline.

---

### 2️⃣ Loading the Dataset

* Loaded the CSV dataset using `pandas.read_csv()`
* Viewed initial rows to understand feature structure

Why:
Understanding the data is essential before applying any model.

---

### 3️⃣ Data Understanding and Preprocessing

* Checked for missing values
* Analyzed data types
* Ensured all features are numerical or encoded

Why:
Logistic Regression requires **clean, numerical input**.

---

### 4️⃣ Selecting Features and Target Variable

* Input features (`X`) → all medical attributes
* Target variable (`y`) → `output`

Why:
This defines the supervised learning mapping.

---

### 5️⃣ Train-Test Split

* Split data into training and testing sets

Why:

* Training data → model learns patterns
* Testing data → evaluates performance on unseen patients

---

### 6️⃣ Feature Scaling

* Applied feature scaling before training

Why:

* Logistic Regression uses gradient-based optimization
* Scaling helps faster and more stable convergence

---

### 7️⃣ Training the Logistic Regression Model

* Created Logistic Regression model using `sklearn`
* Trained the model on training data

Why:
To learn the relationship between health parameters and heart attack risk.

---

### 8️⃣ Making Predictions

* Predicted probabilities
* Converted probabilities into class labels

Why:
Medical decisions require **clear classification output**.

---

### 9️⃣ Model Evaluation

* Evaluated model using classification metrics such as:

  * Accuracy
  * Confusion Matrix
  * Classification Report

Why:
Accuracy alone is not sufficient for medical datasets.

---

## ✅ Final Outcome

* Successfully built a Logistic Regression classification model
* Predicted heart attack risk with reasonable accuracy
* Understood classification workflow end-to-end

---

## 📘 Key Learnings

* Difference between regression and classification
* Logistic Regression theory and sigmoid function
* Importance of feature scaling
* Evaluation metrics for classification problems

---

## 📝 Notes for Future Revision

Remember:

1. Logistic Regression → binary classification
2. Sigmoid outputs probability
3. Threshold converts probability to class
4. Scaling improves performance

If these points are clear, you fully understand **Logistic Regression**.

---

✨ This README is written as **self-revision material**, strictly aligned with the dataset and the notebook implementation.

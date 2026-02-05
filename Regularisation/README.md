# Lasso Regression – Medical Insurance Cost Prediction

This README is written as **complete revision notes for my Lasso Regression notebook**.
If I read this file in the future, I should be able to clearly recall:

* **What problem I was solving**
* **Which dataset I used and why**
* **What Lasso Regression is (theory)**
* **What steps I performed in the notebook**
* **Why each step was necessary**

This file is meant for **self-revision + conceptual clarity**, not just documentation.

---

## 📌 Problem Overview

The goal of this project is to **predict medical insurance charges** for individuals based on their personal and lifestyle attributes using **Lasso Regression**.

Medical insurance cost prediction is a **regression problem** because:

* The target variable (`charges`) is **continuous**
* Multiple factors jointly influence the final cost

---

## 📊 Dataset Information

* **Dataset Name:** Medical Insurance Dataset
* **Source:** Kaggle
* **Link:** [https://www.kaggle.com/datasets/mirichoi0218/insurance](https://www.kaggle.com/datasets/mirichoi0218/insurance)

### Dataset Description

Each row represents one person insured under a medical insurance plan.

### Features Used

| Feature  | Meaning                              |
| -------- | ------------------------------------ |
| age      | Age of the person                    |
| sex      | Gender (male / female)               |
| bmi      | Body Mass Index                      |
| children | Number of children covered           |
| smoker   | Smoking status                       |
| region   | Residential region                   |
| charges  | **Target variable** (insurance cost) |

---

## 🎯 Objective of Using Lasso Regression

Previously, I implemented **Linear Regression** on the same dataset.
In this notebook, the objective was to:

* Understand **regularization**
* See how Lasso Regression improves upon simple Linear Regression
* Observe how **irrelevant features are penalized**

---

## 🧠 Theory: Concepts Used in This Notebook

### 1️⃣ Linear Regression (Base Concept)

Linear Regression models the relationship as:

```
charges = w₁x₁ + w₂x₂ + ... + b
```

Problem with basic Linear Regression:

* Can **overfit** when too many features exist
* Coefficients can become very large

---

### 2️⃣ Regularization (Why It Is Needed)

Regularization adds a **penalty** to the loss function to:

* Reduce overfitting
* Control model complexity
* Improve generalization

---

### 3️⃣ Lasso Regression (L1 Regularization)

Lasso Regression modifies Linear Regression by adding **L1 penalty**:

```
Loss = MSE + λ × Σ|w|
```

Where:

* `MSE` → Mean Squared Error
* `λ (alpha)` → regularization strength
* `|w|` → absolute value of coefficients

#### Key Properties of Lasso:

* Shrinks coefficients
* Can make some coefficients **exactly zero**
* Performs **automatic feature selection**

This makes Lasso very useful when:

* Dataset has many features
* Some features are less important

---

## 🔧 Step-by-Step: What I Did and Why

### 1️⃣ Importing Libraries

Used:

* `pandas` → data handling
* `numpy` → numerical operations
* `matplotlib / seaborn` → visualization
* `sklearn` → model training and evaluation

Why:
These are essential tools for implementing regression models.

---

### 2️⃣ Loading the Dataset

* Loaded the CSV file using `pandas.read_csv()`
* Viewed the dataset to understand structure and columns

Why:
Understanding the data is necessary before modeling.

---

### 3️⃣ Data Preprocessing

Performed preprocessing steps such as:

* Handling categorical variables (encoding)
* Separating features (`X`) and target (`y`)

Why:
Lasso Regression only works with **numerical input**.

---

### 4️⃣ Train-Test Split

* Split the dataset into training and testing sets

Why:

* Training data → model learns patterns
* Testing data → evaluates performance on unseen data

---

### 5️⃣ Feature Scaling

* Applied feature scaling before Lasso Regression

Why scaling is important for Lasso:

* L1 penalty is **scale-sensitive**
* Features with larger values dominate without scaling

---

### 6️⃣ Training the Lasso Regression Model

* Created a Lasso model using `sklearn`
* Selected an appropriate `alpha` value
* Trained the model on training data

Why:

* `alpha` controls regularization strength
* Higher alpha → more shrinkage

---

### 7️⃣ Making Predictions

* Used the trained model to predict insurance charges

Why:
Prediction shows how well the model generalizes.

---

### 8️⃣ Model Evaluation

* Evaluated the model using regression metrics
* Compared performance with Linear Regression

Why:
To understand whether regularization improved the model.

---

### 9️⃣ Observing Coefficients

* Analyzed model coefficients
* Observed which features were reduced or set to zero

Why:
This demonstrates **feature selection behavior of Lasso**.

---

## ✅ Final Outcome

* Built a Lasso Regression model successfully
* Understood impact of regularization
* Learned how Lasso helps control overfitting
* Observed automatic feature selection

---

## 📘 Key Learnings

* Difference between Linear and Lasso Regression
* Importance of feature scaling
* Role of regularization in ML
* How model complexity affects performance

---

## 📝 Notes for Future Revision

When revisiting this project, remember:

1. Lasso = Linear Regression + L1 penalty
2. Scaling is mandatory
3. Alpha controls feature elimination
4. Lasso helps with interpretability

If you understand these points, you understand **Lasso Regression completely**.

---

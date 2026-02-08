# Employee Turnover Prediction – Supervised ML Assignment 2

This README is written as **complete self-revision notes** for **Supervised Machine Learning – Assignment 2**.
It explains:

* **The exact assignment question**
* **What was expected to be done**
* **What I implemented in `employee_turnover_project.ipynb`**
* **Why each model and step was chosen**
* **The theory behind Logistic Regression and Regularization (L1 & L2)**

If I read this README again in the future, I should be able to **fully recall the assignment, concepts, and solution approach** without re-learning everything.

---

## 📌 Assignment Question (As Given)

A multinational company, **TalentCore Pvt. Ltd.**, is facing a high number of employee resignations, leading to:

* Increased recruitment costs
* Project delays
* Loss of skilled talent

The HR department wants to build an **intelligent Machine Learning system** that can **predict whether an employee is likely to leave the company**, based on factors such as:

* Job satisfaction
* Salary
* Age
* Work-life balance
* Training hours
* Bonuses
* Other work-related attributes

### My Role in the Assignment

I was hired as an **AI/ML Engineer** and asked to:

1. Build a **baseline Logistic Regression model**
2. Improve it using **Regularization techniques (L1 & L2)**
3. **Compare model performances** and recommend the best one

(Assignment reference: Supervised ML – Assignment 2) fileciteturn1file0

---

## 🎯 Objective of the Project

The objective is to solve a **binary classification problem**:

* **Target Variable:** `Employee_Turnover`

  * `1` → Employee left the company
  * `0` → Employee stayed

The goal is to **predict employee attrition in advance**, so the company can take preventive actions.

---

## 📂 Files in This Folder

```
assignment/
 ├── employee_turnover_project.ipynb   # Main implementation
 ├── Employee_Turnover_Dataset.csv     # Dataset (uploaded on GitHub)
 └── README.md                         # This revision file
```

---

## 📊 Dataset Description

The dataset contains **900 rows and 15 features**, representing realistic employee information used in corporate environments.

### Key Features Used

| Feature                                 | Description                         |
| --------------------------------------- | ----------------------------------- |
| Job_Satisfaction                        | Employee satisfaction level         |
| Performance_Rating                      | Performance score                   |
| Years_At_Company                        | Years worked in the company         |
| Work_Life_Balance                       | Work–life balance rating            |
| Distance_From_Home                      | Distance between home and workplace |
| Monthly_Income                          | Monthly salary                      |
| Education_Level                         | Education qualification             |
| Age                                     | Age of employee                     |
| Num_Companies_Worked                    | Previous companies worked           |
| Employee_Role                           | Encoded job role                    |
| Department                              | Encoded department                  |
| Annual_Bonus                            | Yearly bonus                        |
| Training_Hours                          | Training hours attended             |
| Annual_Bonus_Squared                    | Engineered feature (bonus²)         |
| Annual_Bonus_Training_Hours_Interaction | Interaction feature                 |
| Employee_Turnover                       | **Target variable**                 |

fileciteturn1file0

---

## 🧠 Theory: Concepts Used in This Assignment

### 1️⃣ Logistic Regression (Baseline Model)

Logistic Regression is used for **binary classification** problems.
It predicts the **probability** of an event using the sigmoid function:

```
P(y=1) = 1 / (1 + e^-(w·x + b))
```

* Output lies between 0 and 1
* A threshold (usually 0.5) converts probability to class

Why Logistic Regression:

* Interpretable
* Works well for structured/tabular data
* Ideal as a baseline model

---

### 2️⃣ Problem of Overfitting

With many features:

* Model can fit noise
* Coefficients can grow large
* Generalization performance reduces

This motivates **regularization**.

---

### 3️⃣ L1 Regularization (Lasso Logistic Regression)

Adds absolute penalty on coefficients:

```
Loss = LogLoss + λ × Σ|w|
```

Key effects:

* Shrinks coefficients
* Can set some coefficients to **zero**
* Performs **feature selection**

---

### 4️⃣ L2 Regularization (Ridge Logistic Regression)

Adds squared penalty on coefficients:

```
Loss = LogLoss + λ × Σw²
```

Key effects:

* Shrinks coefficients smoothly
* Keeps all features
* Reduces multicollinearity

---

## 🔧 Step-by-Step: What I Did and Why

### 1️⃣ Imported Required Libraries

Used:

* `pandas`, `numpy` → data handling
* `matplotlib`, `seaborn` → visualization
* `sklearn` → model training, scaling, evaluation

Why:
These libraries form the standard supervised ML pipeline.

---

### 2️⃣ Loaded the Dataset

* Loaded the employee dataset from GitHub
* Displayed initial rows to understand structure

Why:
Understanding feature types and target variable is critical.

---

### 3️⃣ Data Preprocessing

* Checked for missing values
* Ensured all categorical features were encoded
* Used engineered features already present in the dataset

Why:
Logistic Regression requires clean numerical data.

---

### 4️⃣ Feature–Target Split

* `X` → employee-related features
* `y` → `Employee_Turnover`

Why:
Defines supervised learning mapping.

---

### 5️⃣ Train–Test Split

* Split data into training and testing sets

Why:
Allows evaluation on unseen employee data.

---

### 6️⃣ Feature Scaling

* Applied scaling before training

Why:

* Logistic Regression is scale-sensitive
* Regularization penalties depend on feature magnitude

---

### 7️⃣ Baseline Logistic Regression Model

* Trained a standard Logistic Regression model
* Evaluated performance

Why:
Acts as a reference model for comparison.

---

### 8️⃣ Logistic Regression with L1 Regularization

* Trained model with L1 penalty
* Observed coefficient reduction and feature elimination

Why:
Helps identify important employee factors.

---

### 9️⃣ Logistic Regression with L2 Regularization

* Trained model with L2 penalty
* Compared stability and performance

Why:
Provides better generalization in many real-world cases.

---

### 🔟 Model Comparison and Recommendation

* Compared baseline, L1, and L2 models
* Evaluated using classification metrics

Why:
Final recommendation must be based on performance, not theory alone.

---

## ✅ Final Outcome

* Built three classification models
* Observed impact of regularization
* Identified best-performing model for employee turnover prediction

---

## 📘 Key Learnings

* How Logistic Regression works for HR analytics
* Why regularization is critical
* Difference between L1 and L2 penalties
* Importance of feature engineering

---

## 📝 Notes for Future Revision

Remember:

1. Employee turnover = binary classification
2. Logistic Regression is the baseline
3. L1 → feature selection
4. L2 → coefficient stabilization
5. Scaling is mandatory

If these are clear, you have fully revised **Assignment 2**.

---

✨ This README is written as **exam-ready + self-revision documentation**, strictly aligned with the assignment question and the notebook implementation.

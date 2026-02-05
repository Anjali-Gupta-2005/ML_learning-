# Linear Regression – Insurance Charges Prediction

## 📊 Dataset Used

* **Dataset:** Medical Insurance Dataset
* **Source:** Kaggle
* **Link:** [https://www.kaggle.com/datasets/mirichoi0218/insurance](https://www.kaggle.com/datasets/mirichoi0218/insurance)

The dataset contains information about people and their medical insurance charges.

### Columns Used in This Notebook

| Column   | Description                             |
| -------- | --------------------------------------- |
| age      | Age of the person                       |
| sex      | Gender (male / female)                  |
| bmi      | Body Mass Index                         |
| children | Number of children                      |
| smoker   | Whether the person smokes (yes / no)    |
| charges  | Insurance charges (**target variable**) |

> Note: The `region` column is **explicitly dropped** and not used in this model.

---

## 🎯 Objective of This Notebook

To understand and implement **Linear Regression using sklearn** and predict **insurance charges** based on selected input features.

This notebook focuses on:

* Basic data loading
* Simple visualization
* Manual encoding
* Model training
* Prediction
* Evaluation using R² and Adjusted R²

---

## 🧠 Step-by-Step Explanation (As Done in the Notebook)

### 1️⃣ Importing Libraries

```python
import pandas as pd
import seaborn as sns
```

* `pandas` is used for data handling
* `seaborn` is used for visualization

---

### 2️⃣ Loading the Dataset

```python
insurance_data = pd.read_csv("insurance.csv")
```

* The dataset is loaded into a pandas DataFrame
* The full dataset is displayed to understand its structure

---

### 3️⃣ Data Visualization

```python
sns.scatterplot(x=insurance_data["bmi"], y=insurance_data["charges"], hue=insurance_data["smoker"])
```

**Purpose:**

* To visually understand the relationship between `bmi` and `charges`
* To see the effect of smoking status

**Observation from plot:**

* Smokers (`yes`) generally have **higher insurance charges** compared to non-smokers

---

### 4️⃣ Selecting Input and Output Variables

```python
X = insurance_data.drop(columns=["charges", "region"])
y = insurance_data["charges"]
```

* `X` contains input features
* `y` contains the target variable (`charges`)
* `region` is removed and not used

---

### 5️⃣ Encoding Categorical Variables

```python
X["sex"] = X["sex"].map({"female": 1, "male": 0})
X["smoker"] = X["smoker"].map({"yes": 1, "no": 0})
```

* Since Linear Regression works with numerical values:

  * `sex` is manually encoded
  * `smoker` is manually encoded

---

### 6️⃣ Train-Test Split

```python
from sklearn.model_selection import train_test_split
```

* The dataset is split into:

  * `X_train`, `X_test`
  * `y_train`, `y_test`
* Split is approximately **80% training and 20% testing**

---

### 7️⃣ Training the Linear Regression Model

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
```

* A Linear Regression model is created using sklearn
* The model is trained on training data

---

### 8️⃣ Making Predictions

```python
y_pred = model.predict(X_test)
```

* The trained model predicts insurance charges for test data
* Predicted values are stored in `y_pred`

---

### 9️⃣ Evaluating the Model (R² Score)

```python
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("r-squared : ", r2)
```

* **R² score** measures how well the model explains variance in the data
* Value closer to 1 indicates better performance

---

### 🔟 Adjusted R² Calculation

```python
n = X_test.shape[0]
p = X_test.shape[1]
adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
print("adjusted r squared value :", adjusted_r2)
```

* Adjusted R² is calculated manually
* It accounts for the number of features used
* Helps check if adding features is actually useful

---

## ✅ What This Notebook Covers

✔ Simple Linear Regression workflow
✔ Manual categorical encoding
✔ Visualization-based insight
✔ Model training & prediction
✔ R² and Adjusted R² evaluation

---

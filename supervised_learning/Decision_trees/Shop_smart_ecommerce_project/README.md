# ShopSmart Purchase Prediction using Decision Tree

## Project Overview

This project is based on a supervised machine learning assignment where the goal is to predict whether a visitor on an e-commerce website will make a purchase or not.

The company in the assignment is called **ShopSmart**. The company collects data from user browsing sessions and wants to understand visitor behaviour better. Every user session contains information like:

* Number of pages visited
* Time spent on pages
* Bounce rate
* Exit rate
* Visitor type
* Month of visit
* Traffic source
* Weekend visits

Using this information, the task is to build a **Decision Tree based classification model** that predicts:

* `Revenue = 1` → Customer purchased something
* `Revenue = 0` → Customer did not purchase anything

The assignment specifically mentions that the dataset is imbalanced, so the evaluation metric used is:

* **F1 Score**

The assignment also requires:

* Data preprocessing
* Feature transformation
* Decision Tree pruning
* Model evaluation
* Hyperparameter tuning

---

# Assignment Understanding

The dataset contains around 12,330 browsing sessions.

Each row represents one unique user session.

The target column is:

```python
Revenue
```

This column tells whether the visitor purchased something or not.

The main objective of the project is:

> Predict customer purchase intention using browsing behaviour.

---

# Libraries Used and Why

## 1. Pandas

```python
import pandas as pd
```

### Why it is used

Pandas is used for handling datasets in tabular form.

### What it does in this project

* Reads CSV file
* Stores dataset in DataFrame format
* Helps in selecting columns
* Helps in preprocessing

### Why this line is important

Without Pandas, it would be very difficult to manipulate structured data efficiently.

---

## 2. NumPy

```python
import numpy as np
```

### Why it is used

NumPy is used for numerical computations.

### What it does here

Even though NumPy is not directly used heavily in the notebook, many machine learning libraries internally depend on NumPy arrays.

### Why it is imported

It is considered a standard practice in machine learning projects.

---

## 3. train_test_split

```python
from sklearn.model_selection import train_test_split
```

### Why it is used

Machine learning models should not be trained and tested on the same data.

### What it does

This function divides the dataset into:

* Training data
* Testing data

### Why it is important

* Training data teaches the model
* Testing data checks model performance on unseen data

Without splitting, the model may memorize the data instead of learning patterns.

---

## 4. ColumnTransformer

```python
from sklearn.compose import ColumnTransformer
```

### Why it is used

Different preprocessing methods are required for:

* Numerical columns
* Categorical columns

### What it does

It applies different transformations to different column types.

### Why it is important

* Numerical features need scaling
* Categorical features need encoding

ColumnTransformer combines both operations in one clean pipeline.

---

## 5. StandardScaler

```python
from sklearn.preprocessing import StandardScaler
```

### Why it is used

Numerical values in the dataset may have very different ranges.

Example:

* ProductRelated_Duration may be in thousands
* BounceRates may be between 0 and 1

### What StandardScaler does

It standardizes numerical data using:

genui{"math_block_widget_always_prefetch_v2":{"content":"z = \frac{x-\mu}{\sigma}"}}

Where:

* `x` = original value
* `μ` = mean
* `σ` = standard deviation

### Why scaling is useful

Scaling helps maintain balanced feature contribution.

Although Decision Trees are not highly sensitive to scaling, preprocessing makes the pipeline cleaner and more generalizable.

---

## 6. OneHotEncoder

```python
from sklearn.preprocessing import OneHotEncoder
```

### Why it is used

Machine learning models cannot directly understand text categories.

Example categorical columns:

* Month
* VisitorType

### What OneHotEncoder does

It converts categories into binary columns.

Example:

| VisitorType       | Returning_Visitor | New_Visitor |
| ----------------- | ----------------- | ----------- |
| Returning_Visitor | 1                 | 0           |
| New_Visitor       | 0                 | 1           |

### Why `handle_unknown="ignore"` is used

```python
OneHotEncoder(handle_unknown="ignore")
```

This prevents errors when the test data contains categories not seen during training.

---

## 7. Pipeline

```python
from sklearn.pipeline import Pipeline
```

### Why it is used

A machine learning workflow contains multiple sequential steps.

Example:

1. Preprocessing
2. Model training
3. Prediction

### What Pipeline does

It connects all steps together into one reusable object.

### Why it is important

* Prevents data leakage
* Makes code cleaner
* Makes tuning easier
* Ensures same preprocessing during training and testing

---

## 8. DecisionTreeClassifier

```python
from sklearn.tree import DecisionTreeClassifier
```

### Why it is used

The assignment specifically requires a Decision Tree based classifier.

### What Decision Tree does

It learns rules from data.

Example:

* If page value is high → likely purchase
* If bounce rate is high → unlikely purchase

The model creates branching conditions until it reaches a prediction.

### Why Decision Trees are useful

* Easy to understand
* Works with numerical and categorical data
* Captures nonlinear relationships
* Requires little feature engineering

---

## 9. Evaluation Metrics

```python
from sklearn.metrics import f1_score, classification_report, confusion_matrix
```

### Why they are used

These functions measure model performance.

---

### F1 Score

```python
f1_score()
```

The dataset is imbalanced.

That means:

* Non-purchase sessions are much higher
* Purchase sessions are fewer

Accuracy alone becomes misleading.

F1 Score balances:

* Precision
* Recall

Formula:

genui{"math_block_widget_always_prefetch_v2":{"content":"F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}"}}

### Why F1 is important here

It gives balanced performance measurement for imbalanced datasets.

---

### Classification Report

```python
classification_report()
```

This gives:

* Precision
* Recall
* F1-score
* Support

for every class.

### Why it is useful

It provides detailed insight into model behaviour.

---

### Confusion Matrix

```python
confusion_matrix()
```

This shows:

* True Positives
* True Negatives
* False Positives
* False Negatives

### Why it is useful

It helps understand where the model is making mistakes.

---

# Step-by-Step Code Explanation

---

# Step 1: Importing Libraries

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
```

### What happens here

All required libraries are imported.

### Why this step exists

Before building a machine learning model, we need:

* Data handling tools
* Preprocessing tools
* Model building tools
* Evaluation tools

This cell prepares the environment for the complete workflow.

---

# Step 2: Loading Dataset

```python
df = pd.read_csv("online_shoppers.csv")
```

### What this line does

Reads the CSV dataset file.

### Output

The dataset gets stored inside:

```python
df
```

which is a Pandas DataFrame.

### Why this step is necessary

Machine learning models need structured input data.

This step loads the browsing session dataset into memory.

---

# Step 3: Separating Features and Target

```python
X = df.drop(columns=["Revenue"])
y = df["Revenue"].astype(int)
```

---

## Understanding `X`

```python
X = df.drop(columns=["Revenue"])
```

### What it does

Removes the target column.

### Why

The model should not use the answer column while learning.

### Result

`X` contains all input features like:

* ProductRelated
* BounceRates
* VisitorType
* Month
* ExitRates

etc.

---

## Understanding `y`

```python
y = df["Revenue"].astype(int)
```

### What it does

Extracts the target column.

### Why `.astype(int)` is used

The target may originally be:

* True / False

Machine learning algorithms work better with numeric labels.

So:

* True → 1
* False → 0

### Why this step is important

This creates:

* Input data (`X`)
* Output labels (`y`)

which are required for supervised learning.

---

# Step 4: Identifying Numerical and Categorical Features

```python
num_features = X.select_dtypes(include=["int64", "float64"]).columns
cat_features = X.select_dtypes(include=["object", "category"]).columns
```

---

## Numerical Features

```python
num_features = X.select_dtypes(include=["int64", "float64"]).columns
```

### What it does

Selects columns having numeric data types.

Examples:

* BounceRates
* ExitRates
* ProductRelated_Duration

### Why this is needed

Numerical columns require scaling.

---

## Categorical Features

```python
cat_features = X.select_dtypes(include=["object", "category"]).columns
```

### What it does

Selects text-based categorical columns.

Examples:

* Month
* VisitorType

### Why this is needed

Categorical columns require encoding.

---

# Step 5: Splitting Data into Training and Testing Sets

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

---

## Understanding `test_size=0.2`

```python
test_size=0.2
```

### Meaning

* 80% data used for training
* 20% data used for testing

### Why this ratio is common

It provides enough training data while still keeping unseen data for evaluation.

---

## Understanding `random_state=42`

```python
random_state=42
```

### Why it is used

Ensures reproducibility.

This means every time the code runs, the same split is generated.

### Why reproducibility matters

Without this, results may vary every run.

---

## Understanding `stratify=y`

```python
stratify=y
```

### Why it is extremely important

The dataset is imbalanced.

Stratification ensures:

* Same class distribution in train data
* Same class distribution in test data

### Example

If purchase sessions are 15% overall, stratification preserves that ratio.

### Why this improves the model

Without stratification, one split may contain very few positive samples.

That would make evaluation unreliable.

---

# Step 6: Building the Preprocessing Pipeline

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ]
)
```

---

## Numerical Transformation

```python
("num", StandardScaler(), num_features)
```

### What it does

Applies scaling to numerical columns.

### Why

Normalizes feature values.

---

## Categorical Transformation

```python
("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
```

### What it does

Applies One Hot Encoding to categorical columns.

### Why

Converts text categories into numerical format.

---

## Why ColumnTransformer is powerful

It allows multiple preprocessing operations in one unified structure.

This makes the machine learning workflow cleaner and easier to manage.

---

# Step 7: Creating the Decision Tree Model

```python
dt = DecisionTreeClassifier(
    max_depth=6,
    min_samples_leaf=30,
    class_weight="balanced",
    random_state=42
)
```

This is one of the most important parts of the project.

---

## Understanding `max_depth=6`

```python
max_depth=6
```

### What it controls

Maximum depth of the decision tree.

### Why it is important

A very deep tree memorizes training data.

This causes:

* Overfitting
* Poor generalization

### Why depth 6 is used

It acts as pruning.

The assignment specifically asks for pruning.

### How pruning helps

* Reduces overfitting
* Makes rules simpler
* Improves performance on unseen data

---

## Understanding `min_samples_leaf=30`

```python
min_samples_leaf=30
```

### What it controls

Minimum number of samples required in a leaf node.

### Why it matters

Without this restriction:

* Tree may create tiny leaves
* Tiny leaves memorize noise

### Why 30 is chosen

It smooths decision boundaries and prevents overly specific rules.

This is another pruning technique.

---

## Understanding `class_weight="balanced"`

```python
class_weight="balanced"
```

### Why this is extremely important

The dataset is imbalanced.

Purchase sessions are fewer.

Without balancing:

* Model may predict mostly non-purchases
* Minority class may be ignored

### What balanced weighting does

It automatically gives higher importance to minority class samples.

### Why this improves F1 score

It helps improve recall for purchase predictions.

---

## Understanding `random_state=42`

Again used for reproducibility.

Ensures consistent tree generation.

---

# Step 8: Creating the Full Pipeline

```python
pipe = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", dt)
    ]
)
```

---

## What this pipeline does

The pipeline combines:

1. Preprocessing
2. Model training

into one structure.

---

## Why this is very important

Without pipeline:

* Manual preprocessing becomes repetitive
* Risk of data leakage increases
* Code becomes messy

---

## Benefits of pipeline

* Cleaner workflow
* Easier hyperparameter tuning
* Consistent preprocessing
* Better project organization

---

# Step 9: Training the Model

```python
pipe.fit(X_train, y_train)
```

---

## What happens internally

### Step 1

Training data is preprocessed.

* Numerical columns scaled
* Categorical columns encoded

### Step 2

Processed data is passed to Decision Tree.

### Step 3

The model learns decision rules.

---

## Why training is important

This is the learning phase.

The model identifies patterns between:

* Browsing behaviour
* Purchase outcomes

---

# Step 10: Making Predictions

```python
y_pred = pipe.predict(X_test)
```

---

## What this does

Uses the trained pipeline to predict purchase behaviour on unseen test data.

### Important detail

The pipeline automatically preprocesses the test data before prediction.

This is why pipelines are extremely useful.

---

# Step 11: Evaluating the Model

```python
print("F1 score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

---

## F1 Score

Measures balance between:

* Precision
* Recall

### Why assignment focuses on F1

Because the dataset is imbalanced.

F1 gives a more reliable measure than accuracy.

---

## Classification Report

Shows:

* Precision
* Recall
* F1-score
* Support

for each class.

This helps analyze performance deeply.

---

## Confusion Matrix

Shows:

* Correct predictions
* Incorrect predictions

This helps understand:

* False alarms
* Missed purchase predictions

---

# Hyperparameter Tuning Section

The notebook also contains a hyperparameter tuning step.

This improves model performance systematically.

---

# Step 12: Importing GridSearchCV

```python
from sklearn.model_selection import GridSearchCV
```

### Why it is used

Instead of manually trying different parameters, GridSearchCV automates the search process.

---

# Step 13: Creating Parameter Grid

```python
param_grid = {
    "model__max_depth": [4, 6, 8],
    "model__min_samples_leaf": [20, 30, 50]
}
```

---

## Understanding `model__`

The model exists inside a pipeline.

So parameters are accessed using:

```python
model__parameter_name
```

---

## Why these parameters are tuned

### max_depth

Controls tree complexity.

### min_samples_leaf

Controls minimum samples in leaves.

Both directly affect:

* Overfitting
* Generalization
* F1 score

---

# Step 14: Creating GridSearchCV Object

```python
grid = GridSearchCV(
    pipe,
    param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1
)
```

---

## Understanding `scoring="f1"`

### Why F1 scoring is used

Assignment benchmark is based on F1 score.

So tuning optimizes directly for F1.

---

## Understanding `cv=5`

```python
cv=5
```

### What it means

5-fold cross validation.

### How it works

Data is divided into 5 parts.

The model trains and validates multiple times.

### Why this is useful

Produces more reliable performance estimation.

---

## Understanding `n_jobs=-1`

```python
n_jobs=-1
```

### What it does

Uses all CPU cores.

### Why it helps

Speeds up hyperparameter tuning.

---

# Step 15: Training Grid Search

```python
grid.fit(X_train, y_train)
```

### What happens

GridSearchCV trains many models using different parameter combinations.

Then it selects the best performing model.

---

# Step 16: Printing Best Results

```python
print("Best F1:", grid.best_score_)
print("Best params:", grid.best_params_)
```

---

## What these lines show

### `best_score_`

Highest cross-validated F1 score.

### `best_params_`

Best parameter combination found during tuning.

---

# Important Machine Learning Concepts Used

---

# 1. Supervised Learning

The model learns using:

* Input features
* Known output labels

This is called supervised learning.

---

# 2. Classification

The target has two classes:

* Purchase
* No Purchase

So this is a classification problem.

---

# 3. Imbalanced Dataset

One class appears much more than the other.

This affects model learning.

That is why:

* F1 score
* Class balancing
* Stratification

are important.

---

# 4. Overfitting

A model that memorizes training data performs poorly on unseen data.

This project reduces overfitting using:

* max_depth
* min_samples_leaf
* pruning

---

# 5. Pruning

Pruning reduces unnecessary complexity in the decision tree.

This improves generalization.

The assignment specifically required pruning.

---

# Final Workflow Summary

The complete workflow of the project is:

1. Import libraries
2. Load dataset
3. Separate features and target
4. Identify numerical and categorical columns
5. Split data into train and test sets
6. Build preprocessing pipeline
7. Create Decision Tree model
8. Apply pruning techniques
9. Build complete ML pipeline
10. Train the model
11. Make predictions
12. Evaluate performance using F1 score
13. Perform hyperparameter tuning
14. Find best model parameters

---

# Key Learning Outcomes from this Project

After completing this project, the following concepts become clear:

* How supervised learning works
* How classification models work
* Why preprocessing is important
* How categorical encoding works
* Why train-test splitting is required
* Importance of stratification
* Why pipelines are useful
* How Decision Trees make decisions
* What overfitting means
* How pruning improves Decision Trees
* Why F1 score is used for imbalanced datasets
* How hyperparameter tuning improves performance
* How GridSearchCV works

---

# Conclusion

This project successfully builds a Decision Tree based machine learning model to predict online purchase intention.

The project follows a proper machine learning workflow including:

* Data preprocessing
* Feature transformation
* Model building
* Pruning
* Pipeline creation
* Evaluation
* Hyperparameter tuning

Special attention is given to the imbalanced nature of the dataset using:

* F1 score
* Stratification
* Balanced class weights

The final solution demonstrates how machine learning can help e-commerce companies better understand customer behaviour and improve business decision making.

---

# Files Included

* `shop_smart.ipynb` → Complete notebook implementation
* `online_shoppers.csv` → Dataset
* `README.md` → Detailed project explanation
* `Supervised ML Assignment4.pdf` → Assignment problem statement

---

# Reference

Assignment description taken from the uploaded assignment PDF. fileciteturn0file0L1-L40

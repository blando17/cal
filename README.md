#  Calories Burnt Predictor

This machine learning project predicts the number of calories burnt during physical activity based on user metrics like age, weight, heart rate, duration, and more, using the **XGBoost Regressor**.

---

##  Project Overview

Calorie estimation helps monitor fitness and energy expenditure. This project combines data from two sources (`calories.csv` and `exercise.csv`) to build a regression model that accurately predicts calories burnt based on biometric and exercise-related features.

---

## Technologies Used

| Tool/Library         | Purpose                              |
|----------------------|--------------------------------------|
| Python               | Programming language                 |
| Pandas & NumPy       | Data manipulation                    |
| Seaborn & Matplotlib | Data visualization                   |
| Scikit-learn         | Preprocessing and evaluation         |
| XGBoost              | Regression modeling                  |

---

##  Dataset Description

- Total samples: **15,000**
- Combined from:
  - `exercise.csv` (metrics like Age, Duration, Heart Rate, etc.)
  - `calories.csv` (target label)
- Features:
  - `Gender`, `Age`, `Height`, `Weight`, `Duration`, `Heart_Rate`, `Body_Temp`
- Target:
  - `Calories`

There are **no missing values** in the dataset.

---

##  Data Preprocessing

- **Concatenated** exercise and calorie datasets.
- **Label Encoding**:
  - `Gender`: male → 0, female → 1
- Dropped: `User_ID` column (not useful for prediction)

```python
data.replace({"Gender": {'male': 0, 'female': 1}}, inplace=True)
X = data.drop(columns=['User_ID', 'Calories'])
Y = data['Calories']
```

---

##  Exploratory Data Analysis

- Countplot of Gender distribution
- Distribution plots for:
  - Age
  - Height
  - Weight

```python
sns.countplot(x=data['Gender'])
sns.displot(x=data['Age'])
sns.displot(x=data['Height'])
sns.displot(x=data['Weight'])
```

---

##  Model Building

### Model: XGBoost Regressor

```python
from xgboost import XGBRegressor
model = XGBRegressor()
model.fit(X_train, Y_train)
```

Data Split:
```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
```

---

##  Model Evaluation

- **Training Set**:
  - MAE: `0.93`
  - MSE: `1.67`

- **Testing Set**:
  - MAE: `1.48`
  - MSE: `4.71`

Evaluation code:
```python
metrics.mean_absolute_error(Y_test, model.predict(X_test))
metrics.mean_squared_error(Y_test, model.predict(X_test))
```

---

##  Predictive System Example

```python
input_data = (0, 68, 190.0, 94.0, 29.0, 105.0, 40.8)
input_np = np.asarray(input_data).reshape(1, -1)
prediction = model.predict(input_np)
print(prediction)
```

Output: `[236.13]` calories

---

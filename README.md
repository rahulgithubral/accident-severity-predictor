# 🚑 Accident Severity Prediction & Analysis using Machine Learning

This project analyzes **170,000+ UK road accident records** to identify patterns associated with accident severity and predict whether an accident is **Slight, Serious, or Fatal**.

The project combines **Exploratory Data Analysis (EDA), data preprocessing, class-imbalance analysis, and machine learning** to understand accident-severity patterns and evaluate predictive performance using appropriate metrics.

---

## 🎯 Project Objective

The objective of this project is to use real-world road accident data to:

- Explore patterns and relationships associated with accident severity.
- Analyze the distribution and characteristics of different accident-severity classes.
- Identify the impact of road, vehicle, environmental, and time-related factors.
- Build and compare multiclass classification models.
- Evaluate models using metrics that account for severe class imbalance.
- Improve detection of minority **Serious** and **Fatal** accident classes.

---

## 📂 Project Contents

- `Accident_severity_prediction.ipynb` — Complete Google Colab notebook containing:
  - Exploratory Data Analysis (EDA)
  - Data preprocessing
  - Feature selection and encoding
  - Class distribution analysis
  - Machine learning model development
  - Model comparison
  - Confusion matrix analysis
  - Feature-importance analysis
  - Class-imbalance analysis

---

## 📊 Features Analyzed

The analysis considers factors including:

- Weather conditions
- Light conditions
- Road surface conditions
- Speed limit
- Time of accident
- Vehicle type
- Urban/Rural indicator
- Other road, vehicle, environmental, and accident-related factors

---

## 🔍 Exploratory Data Analysis

The dataset was explored to identify relationships between accident severity and different road, environmental, vehicle, and time-related factors.

Key analysis included:

- Accident severity distribution
- Feature distributions
- Relationships between environmental conditions and severity
- Comparison of characteristics across severity classes
- Identification of class imbalance

The dataset contains a **highly imbalanced target variable**, with Slight accidents substantially outnumbering Serious and Fatal accidents.

---

## 🤖 Machine Learning Models

The following multiclass classification models were developed and compared:

- **K-Nearest Neighbors (KNN)**
- **Random Forest**
- **XGBoost**
- **Tuned XGBoost**

Models were evaluated using:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Confusion Matrix

Macro-averaged metrics were emphasized because accuracy alone can be misleading when minority classes are underrepresented.

---

## 📈 Model Performance

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|------|----------|----------------|--------------|----------|
| KNN | 83.66% | 0.380 | 0.342 | 0.333 |
| Random Forest | 85.28% | 0.413 | 0.335 | 0.312 |
| XGBoost | 58.06% | 0.388 | 0.522 | 0.361 |
| Tuned XGBoost | 59.36% | 0.389 | 0.516 | **0.367** |

### Key Finding

KNN and Random Forest achieved approximately **84–85% accuracy**, but their low macro-recall and macro-F1 scores revealed poor performance on minority severity classes.

The tuned XGBoost model achieved the highest **Macro F1-score of 0.367** and approximately **51% recall for Fatal accidents**, providing substantially better minority-class detection.

---

## 🧠 Key Insights

- Accuracy alone was misleading because of severe class imbalance.
- Random Forest achieved the highest raw accuracy but performed poorly on minority classes.
- Macro-averaged metrics provided a more meaningful comparison across all severity categories.
- XGBoost improved detection of Serious and Fatal accidents compared with the baseline models.
- Feature-importance analysis was used to identify variables contributing to severity predictions.

---

## ⚙️ Data Preprocessing

The preprocessing workflow included:

- Handling missing values
- Feature selection
- Categorical variable encoding
- Train-test splitting
- Class-balancing techniques for XGBoost
- Model evaluation using class-balanced metrics

---

## 🧠 Dataset

**Source:** UK Road Safety — Kaggle

**Dataset:** UK Road Safety: Traffic Accidents and Vehicles

**File:** `Accidents_UK.csv`

The dataset contains UK road accident records covering accident, road, vehicle, environmental, and time-related characteristics.

---

## 🛠️ Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **XGBoost**
- **Google Colab**

---

## 🧪 How to Run the Notebook in Google Colab

1. Open `Accident_severity_prediction.ipynb` in Google Colab.
2. Download `Accidents_UK.csv` from Kaggle.
3. Upload the dataset to Colab.
4. Run the notebook cells sequentially.

```python
from google.colab import files
uploaded = files.upload()

import pandas as pd
import io

df = pd.read_csv(io.BytesIO(next(iter(uploaded.values()))))

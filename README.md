# 🚑 Accident Severity Prediction using Machine Learning

This project predicts the **severity of road accidents** in the UK—classified as *Slight*, *Serious*, or *Fatal*—based on conditions like weather, light, road surface, and time. It explores the dataset using EDA, processes the data, and compares multiple machine learning models to identify the most accurate classifier.

> ✅ This version was prepared and published individually for portfolio and resume purposes.

---

## 📂 Project Contents

- `Accident_severity_prediction.ipynb` — Colab notebook with:
  - Exploratory Data Analysis (EDA)
  - Data preprocessing
  - ML model training and evaluation

---

## 📊 Features Used

- Weather conditions  
- Light conditions  
- Road surface  
- Speed limit  
- Time of day  
- Vehicle type  
- Urban/Rural indicator

---

## ⚙️ Machine Learning Models Used

- **K-Nearest Neighbors (KNN)**
- **Logistic Regression**
- **Random Forest**
- **Decision Tree**

Each model is evaluated using:
- Accuracy Score
- Confusion Matrix
- Classification Report

---

## 📈 Performance

- **Best Model:** Random Forest  
- **Accuracy:** ~82%  
- Models were tuned using default hyperparameters for quick comparison.

---

## 🧠 Dataset

- Source: [UK Road Safety - Kaggle](https://www.kaggle.com/datasets/saurabhshahane/uk-road-safety-accidents-and-vehicles)
- File used: `Accidents_UK.csv`
- Preprocessing includes:
  - Handling missing values
  - Label encoding
  - Feature selection

---

## 🧪 How to Run the Notebook in Google Colab

1. Click **Open in Colab** (or open the `.ipynb` manually in Colab)
2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/uk-road-safety-accidents-and-vehicles)
3. In Colab, upload the dataset using:
   ```python
   from google.colab import files
   uploaded = files.upload()

   import pandas as pd
   import io
   df = pd.read_csv(io.BytesIO(next(iter(uploaded.values()))))

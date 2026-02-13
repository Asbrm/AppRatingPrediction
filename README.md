# Google Play Store App Rating Prediction

This project focuses on predicting the success of Android applications on the Google Play Store using advanced Machine Learning techniques. The analysis is performed using both **Regression** (to predict exact ratings) and **Classification** (to predict if an app's rating is $\ge$ 4.0).

## 📌 Project Overview

The main objective is to understand what drives an app's success by analyzing various features such as category, reviews, size, installs, and price.

### Key Features:
* **Data Cleaning:** Handling missing values, duplicates, and converting data types.
* **Feature Engineering:** Extracting sentiment polarity and subjectivity from user reviews.
* **Model Comparison:** Implementing and comparing multiple algorithms to find the best predictor.

## 📂 Dataset

The project uses the publicly available **Google Play Store Apps** dataset:
1.  **`googleplaystore.csv`**: Contains details of the applications (Category, Rating, Reviews, Size, etc.).
2.  **`googleplaystore_user_reviews.csv`**: Contains user reviews used for sentiment analysis.

## 🧠 Models Implemented

I have implemented and compared a wide range of models for both tasks:

### 1. Regression Models
* **Goal:** Predict the exact numerical rating (1.0 - 5.0).
* **Algorithms:**
    * Random Forest Regressor
    * Extra Trees Regressor
    * Gradient Boosting Regressor
    * Artificial Neural Network (ANN) using PyTorch

### 2. Classification Models
* **Goal:** Classify apps as "High Rated" (Rating $\ge$ 4.0) or "Low Rated".
* **Algorithms:**
    * XGBoost Classifier
    * LightGBM Classifier
    * CatBoost Classifier
    * Support Vector Classifier (SVC)

## ⚙️ Installation & Requirements

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/AppRatingPrediction.git](https://github.com/your-username/AppRatingPrediction.git)
    cd AppRatingPrediction
    ```

2.  **Install the required Python libraries:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm catboost torch
    ```

3.  **Run the Notebook:**
    Open the Jupyter Notebook to view the analysis and model training:
    ```bash
    jupyter notebook
    ```

## 📊 Results & Conclusion

The project evaluates models based on metrics such as **Accuracy** (for classification) and **Mean Squared Error (MSE)** (for regression). The tree-based ensemble methods (like Random Forest and XGBoost) generally provided the most robust results for this dataset.

---
*Developed by Ahmed Saad*

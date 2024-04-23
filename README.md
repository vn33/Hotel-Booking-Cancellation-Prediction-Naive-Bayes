# Hotel Booking Cancellation Prediction using Naive Bayes Classifier
![Evaluation Results](https://github.com/vn33/Hotel-Booking-Cancellation-Prediction-Naive-Bayes/blob/master/Evaluation_metrics.png)
## Overview
This project focuses on feature engineering techniques to predict hotel booking cancellations using machine learning. By conducting exploratory data analysis (EDA) and transforming the dataset's features, we aim to preprocess the data for modeling. Additionally, we have applied a Naive Bayes Classifier to enhance our predictive model.

## Problem Statement
The task is to engineer features that effectively capture the factors influencing hotel booking cancellations. This involves handling missing values, transforming categorical and numerical features, scaling the data, and applying the Naive Bayes Classifier. Subsequently, we will evaluate the performance of the model to predict booking cancellations effectively.

## Activities Completed
1. **Exploratory Data Analysis (EDA)**
   - Conducted data quality checks.
   - Treated missing values.
   - Analyzed categorical data.
   - Analyzed numerical data.

2. **Data Transformation**
   - Transformed categorical data using techniques such as One-Hot Encoder, Label Encoder, and Ordinal Encoder to convert categorical variables into a format suitable for machine learning algorithms.
   - Transformed numerical data using the Power transform to handle skewness and ensure a more normal distribution of features.

3. **Data Scaling**
   - Scaled the data to ensure that all features have the same scale, which is crucial for many machine learning algorithms to perform effectively.

4. **Naive Bayes Classifier**
   - Applied the Naive Bayes Classifier to the preprocessed data to enhance the predictive model.

## Model Evaluation Results
**Inference:**
- **Precision:**
  - For canceled bookings (class 1), precision is around 0.47 on both sets, indicating correct predictions around 47% of the time.
  - Precision for non-canceled bookings (class 0) is higher, approximately 0.87, suggesting correct predictions around 87% of the time.
- **Recall:**
  - The model effectively captures around 90% of actual canceled bookings (class 1), with a recall of approximately 0.90 on both sets.
  - However, it misses around 60% of actual non-canceled bookings (class 0), with a recall of about 0.40 on both sets.
- **F1-score:**
  - The F1-score, which balances precision and recall, is higher for class 1 compared to class 0, indicating better performance in predicting canceled bookings.
- **Accuracy:**
  - The overall accuracy of the model is approximately 59% on both training and test sets, indicating the proportion of correctly classified bookings out of all bookings.

**Generalization:**
- The similarity in performance metrics between training and test sets suggests that the model generalizes adequately and does not exhibit significant overfitting or underfitting issues.

In summary, while the model demonstrates effective identification of canceled bookings (class 1), there is room for improvement in correctly identifying non-canceled bookings (class 0). Further refinement may enhance its accuracy, particularly for predictions of non-cancellation.

## Dataset
The dataset used in this project can be found `hotel.csv`.

## Notebook
The Jupyter notebook containing the code for this project can be found Hotel Booking NaiveBayes.ipynb``.

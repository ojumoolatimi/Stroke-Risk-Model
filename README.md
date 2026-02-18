# Stroke Risk Prediction with Decision Trees

I built this project to predict the likelihood of a stroke using machine learning, specifically focusing on decision tree classifiers. The goal was to understand which factors most strongly drive stroke risk predictions and to compare different handling strategies for the highly imbalanced dataset (stroke cases are rare).

The dataset used is the popular [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) from Kaggle (also known as healthcare-dataset-stroke-data.csv), containing 5,110 patient records with features like age, glucose levels, BMI, hypertension, smoking status, and more.

## Key Insights from My Analysis

- Age is by far the strongest predictor — older age dramatically increases predicted stroke risk.
- Average glucose level ranks second — higher levels are strongly linked to higher risk.
- BMI also plays a meaningful role — higher values contribute to elevated risk.
- Other features (smoking status, work type, heart disease, gender, residence type, etc.) have much smaller influence and mainly help fine-tune predictions.

These findings suggest prevention efforts should prioritize **monitoring older adults**, **managing blood sugar**, and **promoting healthy weight**.

#Project Highlights

- Data wrangling and exploratory analysis (missing BMI values imputed with median)
- Handling class imbalance using:
  - No resampling (baseline)
  - RandomUnderSampler (under-sampling the majority class — my main focus)
  - RandomOverSampler (over-sampling the minority class)
- One-hot encoding for categorical variables
- SimpleImputer (median strategy) for missing values
- DecisionTreeClassifier (with random_state=42 for reproducibility)
- Performance evaluation: accuracy, confusion matrix, and comparison across the three versions
- the important features

## Repository Structure

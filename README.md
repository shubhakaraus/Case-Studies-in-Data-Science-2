# Customer Analytics Using Machine Learning

## Overview

This repository contains my Data Science analysis for **Individual Task 2**. The project extends the machine learning analysis conducted in Individual Task 1 by evaluating model reliability, performance with different training set sizes, and subgroup performance.

The analysis focuses on the **Online Shoppers Purchasing Intention** dataset and investigates whether an online shopping session results in a purchase.

Two machine learning classification models are evaluated:

* **Random Forest**
* **Support Vector Machine (SVM)**

Additional analysis was conducted using cross-validation, learning curves, and Fairlearn to provide a more detailed evaluation of model performance and potential subgroup disparities.

## Dataset

### Online Shoppers Purchasing Intention

The dataset contains **12,330 online shopping sessions** with **18 attributes** describing browsing behaviour, page activity, visitor characteristics and purchase outcomes.

The target variable is `Revenue`, which indicates whether a shopping session resulted in a purchase.

**Source:**

https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

The dataset contains:

* 10,422 non-purchasing sessions
* 1,908 purchasing sessions

The target variable is therefore imbalanced, with purchasing sessions representing approximately 15.47% of the observations.

## Machine Learning Methods

Two classification algorithms were evaluated:

* **Random Forest**
* **Support Vector Machine (SVM)**

The data was divided using an **80:20 stratified train-test split**. Stratification was used to maintain a similar proportion of purchasing and non-purchasing sessions in both the training and test sets.

The preprocessing pipeline included:

* Median imputation for numerical variables
* Most-frequent imputation for categorical variables
* Standardisation of numerical features
* One-hot encoding of categorical variables

The Random Forest model used class weighting to account for the imbalanced target classes.

The SVM model used an RBF kernel and class weighting to improve minority-class detection.

## Features Used

The analysis uses behavioural and session-level variables including:

### Numerical Features

* Administrative
* Administrative_Duration
* Informational
* Informational_Duration
* ProductRelated
* ProductRelated_Duration
* BounceRates
* ExitRates
* PageValues
* SpecialDay
* OperatingSystems
* Browser
* Region
* TrafficType

### Categorical Features

* Month
* VisitorType
* Weekend

The target variable is:

* `Revenue`

## Model Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* PR-AUC

Accuracy was not considered sufficient on its own because the target variable is imbalanced.

Precision and recall were examined to understand the ability of the models to identify purchasing sessions. F1-score was used as a balance between precision and recall.

ROC-AUC was used to evaluate the overall ranking ability of the models, while PR-AUC was included because it provides additional information for imbalanced classification problems.

## Cross-Validation

To reduce dependence on a single train-test split, **5-fold Stratified Cross-Validation** was performed.

The same class proportions were maintained across the folds.

The cross-validation analysis examined:

* Mean F1-score
* F1-score standard deviation
* Mean ROC-AUC
* ROC-AUC standard deviation
* Mean precision
* Mean recall

The cross-validation results provide additional information about the stability of model performance across different subsets of the training data.

## Learning Curve Analysis

Learning curves were generated to examine how model performance changes as the training dataset increases.

Training and validation F1-scores were compared using different training sample sizes.

The investigated training sizes were approximately:

* 789 samples
* 2,564 samples
* 4,340 samples
* 6,115 samples
* 7,891 samples

The learning curves help identify:

* Potential overfitting
* Generalisation behaviour
* The effect of additional training data
* Whether validation performance continues to improve as more data is provided

The Random Forest model showed a large gap between training and validation F1-score, indicating behaviour consistent with overfitting.

The SVM validation F1-score improved as more training data was provided and then began to stabilise.

## Fairness and Bias Analysis

Fairness analysis was conducted using the **Fairlearn** library.

The `VisitorType` feature was used to examine whether model performance differed across visitor groups.

The groups included:

* New Visitor
* Returning Visitor
* Other

Performance was evaluated using:

* Recall
* Precision
* F1-score

The analysis also calculated differences between the highest and lowest subgroup performance.

The results are used as a subgroup performance analysis rather than as definitive evidence of discrimination because `VisitorType` is a behavioural variable rather than a protected demographic attribute.

The analysis also considers the effect of subgroup sample sizes when interpreting differences in performance.

## Security, Privacy and Ethical Considerations

Potential risks were considered if the model were deployed in a real-world customer analytics environment.

Important considerations include:

* Protection of customer browsing and behavioural data
* Secure storage and access control
* Minimisation of collected customer information
* Potential privacy risks associated with behavioural data
* Monitoring for changes in customer behaviour over time
* Potential bias in model predictions
* The consequences of false positives and false negatives
* Human oversight of automated predictions

The model should therefore be treated as a decision-support tool rather than an autonomous decision-maker.

## Results

The current test-set results were:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.8877 | 0.6212 | 0.7042 | 0.6601 | 0.9242 | 0.7241 |
| SVM | 0.8560 | 0.5258 | 0.7199 | 0.6077 | 0.8856 | 0.6238 |

The 5-fold cross-validation results were:

| Model | Mean F1 | F1 Std | Mean ROC-AUC | ROC-AUC Std | Mean Precision | Mean Recall |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.6861 | 0.0143 | 0.9311 | 0.0038 | 0.6383 | 0.7431 |
| SVM | 0.6448 | 0.0183 | 0.9059 | 0.0073 | 0.5610 | 0.7595 |

The results show that the two models have different performance characteristics. The Random Forest achieved higher F1-score and ROC-AUC, while the SVM achieved slightly higher recall during cross-validation.

## Visualisations

The analysis produces learning curves for both models.

The repository can contain:

* Random Forest learning curve
* SVM learning curve
* Model performance tables
* Confusion matrices
* Cross-validation results
* Fairness/subgroup performance results

The learning curves are used to visualise the relationship between training-set size and model generalisation.

## Key Findings

The analysis provides several findings beyond the original Task 1 evaluation.

First, **5-fold stratified cross-validation** provides a more robust view of model stability than relying on a single train-test split.

Second, the **Random Forest learning curve** shows a substantial difference between training and validation F1-score. The model achieves very high training performance while validation performance remains considerably lower, which is consistent with overfitting.

Third, the **SVM validation performance improves as more training data is used**, although the improvement becomes smaller at larger training sizes.

Finally, the **Fairlearn analysis identifies differences in performance between VisitorType groups**. These differences should be interpreted cautiously because VisitorType is a behavioural attribute and subgroup sizes can affect the reliability of individual metrics.

## Technologies

* Python
* Jupyter Notebook
* Pandas
* NumPy
* Scikit-learn
* Fairlearn
* Matplotlib
* Seaborn

## Repository Structure

```text
customer-analytics-machine-learning/
│
├── README.md
│
├── online_shoppers_task2_final_cleaned.py
│
├── img/
│   ├── random_forest_learning_curve.png
│   └── svm_learning_curve.png
│
└── ...

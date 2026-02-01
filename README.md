a.py — Data Generation

This file generates a synthetic dataset for tech hiring.
It intentionally adds bias related to gender, race, and their intersection.
The goal is to simulate a biased hiring process so that fairness methods can be tested in a controlled setting.

b.py — Fairness Analysis and Explanation

This file analyses fairness in the hiring data or model predictions.
It calculates fairness metrics such as Disparate Impact (DI) and Equal Opportunity for different demographic groups.

It also uses logistic regression to check whether some groups are still disadvantaged after controlling for qualifications.

In addition, this file includes SHAP explainability. A tree-based model is trained, and SHAP is used to show which features have the most negative impact on each group, helping to explain where bias comes from.

c.py — Models and Debiasing Experiments

This file contains the main modelling experiments.
It trains a baseline model with no fairness constraints, a gender-based adversarial model, and an intersectional adversarial model using a Gradient Reversal Layer (GRL).

The models are compared based on prediction accuracy and fairness results, in order to study the trade-off between accuracy and fairness.

app.py — Interactive Dashboard

This file creates an interactive web application using Streamlit.
It allows users to upload a hiring dataset and automatically run the fairness analysis.

The app reuses functions from b.py to calculate Disparate Impact (DI), Equal Opportunity, odds ratios, and SHAP explanations.
Results are shown using tables and charts, making the bias easier to understand.
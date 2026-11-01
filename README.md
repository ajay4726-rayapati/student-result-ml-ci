# Student Result Prediction ML CI

## Objective
Automatically train, evaluate and test a machine-learning model using GitHub Actions.

## Input features
- Attendance
- Internal marks
- Assignment marks
- Previous score

## Model
StandardScaler + Logistic Regression.

## Run locally or in Colab
Install requirements, run `python train_model.py`, then
run `python -m unittest discover -v`.

The GitHub Actions workflow repeats these steps on pushes
and pull requests targeting main.

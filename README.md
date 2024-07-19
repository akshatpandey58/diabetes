# Diabetes Prediction using Decision Tree Classifier

This repository contains a Python script that performs data preprocessing, model building, training, visualization, and evaluation of a Decision Tree Classifier for predicting diabetes based on a given dataset. The program uses the diabetes dataset and includes a step-by-step implementation of the machine learning pipeline.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Introduction
The goal of this project is to build a Decision Tree Classifier to predict whether a patient has diabetes based on various medical attributes. The script includes:
- Data loading and preprocessing
- Model training and evaluation
- Visualization of the decision tree
- Prediction for new, unseen data

## Requirements
- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Dataset
The dataset used in this project is the diabetes dataset. Ensure the dataset is available in CSV format and properly placed in the working directory. Replace the placeholder path in the script with the actual path to your dataset file.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/diabetes-prediction.git
    ```
2. Change the directory:
    ```bash
    cd diabetes-prediction
    ```
3. Install the required Python packages:
    ```bash
    pip install numpy pandas scikit-learn matplotlib
    ```

## Usage
1. Ensure the dataset is placed in the working directory and replace `'your_dataset.csv'` in the script with the actual dataset filename.

2. Run the script:
    ```bash
    python diabetes_prediction.py
    ```

3. The script will perform the following steps:
    - Load and preprocess the dataset
    - Split the dataset into training and testing sets
    - Train a Decision Tree Classifier on the training set
    - Visualize the trained Decision Tree
    - Evaluate the model's performance on the testing set
    - Predict the outcome for a new, unseen record

## Results
The script will output:
- Number of records and features in the dataset
- Feature names
- Number of records in each category of the 'Outcome' feature
- Information about missing values in the dataset
- Visualization of the Decision Tree
- Classification report and confusion matrix for the test set
- Predicted outcome for a new, unseen record

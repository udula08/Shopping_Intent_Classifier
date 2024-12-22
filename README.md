# Shopping Intent Classifier

This project includes a machine learning classifier preicting on users will make a purchase during an online shopping session based on previous session data.

## Task Overview

- Loading and preprocessing data from csv file.
- Training the k-nearest neighbor model (k=1)
- Evaluating the classifier using sensitivity and specificity metrics.

## Dataset

- Contains 12330 rows of data.
- The first six columns measure the different types of pages users have visited in the session: the Administrative, Informational, and ProductRelated columns measure how many of those types of pages the user visited, and their corresponding _Duration columns measure how much time the user spent on any of those pages.

## Requirements

- Make sure Python is installed.
- Install the scikit-learn package `pip3 install scikit-learn`.

## Implementation

- `load_data(filename)` - This function loads data from a CSV file and splits it into evidence (features) and labels (target variable).
- `train_model(evidence, labels)` - This function initializes and trains a K-Nearest Neighbors (KNN) model using the provided evidence and labels.
- `evaluate(labels, predictions)` - This function calculates and returns two performance metrics (Sensitivity, Specificity).

## Sample Output
```
Correct: 4092
Incorrect: 840
True Positive Rate: 42.76%
True Negative Rate: 90.21%
```
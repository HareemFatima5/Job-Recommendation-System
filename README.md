# Job Recommendation System Improvement

## Overview

This Jupyter Notebook contains the implementation of a job recommendation system designed to categorize resumes into different job categories based on their features. The system uses machine learning techniques, specifically the Random Forest classifier, to predict job categories from resume text data. The dataset is preprocessed, balanced, and transformed using TF-IDF vectorization before training the model.

## Features

- **Data Loading and Exploration**: Loads and explores the resume dataset, including visualizing the distribution of job categories.
- **Data Balancing**: Uses resampling techniques to balance the dataset, ensuring equal representation of all job categories.
- **Feature Engineering**: Transforms text data into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.
- **Model Training**: Trains a Random Forest classifier to predict job categories from resume features.
- **Evaluation**: Evaluates the model's performance using accuracy metrics and a confusion matrix.

## Dataset

The dataset (`clean_resume_data.csv`) contains the following columns:
- `ID`: Unique identifier for each resume.
- `Category`: The job category associated with the resume (e.g., HR, IT, Engineering).
- `Feature`: The text content of the resume.

## Requirements

To run this notebook, you will need the following Python libraries:
- pandas
- scikit-learn
- matplotlib
- seaborn

You can install these libraries using pip:
```bash
pip install pandas scikit-learn matplotlib seaborn
```
## Results
The Random Forest classifier achieved an accuracy of approximately 83.68% on the test set. The classification report and confusion matrix provide detailed insights into the model's performance across different job categories.

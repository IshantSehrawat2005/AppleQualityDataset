# Apple Quality Prediction - Decision Tree Classifier

## Overview

This project uses a **Decision Tree Classifier** to predict the quality of apples based on given features. The dataset is sourced from Kaggle and contains various attributes related to apple quality.

## Dataset

- **Source:** [Kaggle - Apple Quality Dataset](https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality)
- **Description:** The dataset includes multiple features related to the quality of apples, such as color, texture, and firmness.

## Steps Followed

1. **Data Preprocessing**

   - Loaded the dataset using Pandas
   - Checked for missing values and handled them accordingly
   - Scaled the features using `StandardScaler`

2. **Model Training**

   - Used `DecisionTreeClassifier` from `sklearn`
   - Set `max_depth=16` to prevent overfitting
   - Split the dataset into training and testing sets

3. **Feature Selection**

   - Used `feature_importances_` to find important features
   - Selected only features with importance above a threshold

4. **Model Evaluation**

   - Predicted apple quality using the trained model
   - Calculated accuracy using `accuracy_score`
   - Printed the accuracy score for both training and testing sets

## Key Learnings

- How to preprocess and scale dataset features.
- The importance of feature selection in improving model performance.
- How `max_depth` impacts decision tree complexity and overfitting.
- Evaluating models using accuracy scores.

## Future Improvements

- Try using different models such as Random Forest or XGBoost.
- Perform hyperparameter tuning for better accuracy.
- Visualize decision trees for better interpretability.

## Running the Project

1. Clone the repository
2. Install required dependencies: `pip install -r requirements.txt`
3. Run the Python script to train and evaluate the model.

## Contact

For any questions, feel free to reach out!


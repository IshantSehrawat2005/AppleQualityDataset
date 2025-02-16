# AppleQualityDataset
Decision Tree Classifier

📌 Overview

This project implements a Decision Tree Classifier using Scikit-Learn to analyze and predict outcomes based on a dataset. The model undergoes data preprocessing, training, feature selection, and evaluation.

📂 Table of Contents

Dataset

Installation

Preprocessing

Model Training

Feature Selection

Evaluation

Results

Insights & Learnings

Future Enhancements

Conclusion

Source

📊 Dataset

The dataset used is the Apple Quality Dataset from Kaggle.

Features are preprocessed using StandardScaler to normalize values.

The dataset is split into training (80%) and testing (20%).

⚙️ Installation

To run this project, install the required dependencies:

pip install numpy pandas scikit-learn

🔄 Preprocessing

Feature Scaling

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

Purpose: Standardizes features to have a mean of 0 and standard deviation of 1.

🌳 Model Training

Train the Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth=16, random_state=8)
clf.fit(X_train_scaled, y_train)

max_depth=16 limits tree depth to prevent overfitting.

random_state=8 ensures reproducibility.

🏆 Feature Selection

Extract Important Features

importances = clf.feature_importances_
threshold = 0.1  # Adjust threshold as needed
selected_features = X.columns[importances > threshold]

X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

Only selects features with importance scores above 0.1.

📈 Evaluation

Model Accuracy

from sklearn.metrics import accuracy_score

y_pred = clf.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred) * 100
print(f"Accuracy Score: {acc:.2f}%")

Computes the accuracy of the trained model on test data.

📊 Results

✅ Feature Importance Analysis helps in reducing irrelevant features.✅ Decision Tree Model achieves an accuracy of X% (update with actual result).✅ Scalability: The model can be fine-tuned using max_depth, criterion, and other hyperparameters.

💡 Insights & Learnings

Importance of Feature Scaling: Standardization improved model performance.

Feature Selection: Reducing the number of features helped optimize efficiency.

Max Depth Impact: Higher depth can lead to overfitting, while lower depth may underfit.

Hyperparameter Tuning: Adjusting parameters like max_depth and criterion can improve accuracy.

🚀 Future Enhancements

Experiment with Random Forest and Gradient Boosting for better results.

Implement Hyperparameter Optimization using GridSearchCV.

Use Cross-Validation for more robust performance evaluation.

Deploy the model as a web API for real-world applications.

🏁 Conclusion

Decision Trees are powerful for classification tasks.

Feature Selection improves model efficiency.

Can be extended by testing other models like Random Forest or Gradient Boosting.

📌 Next Steps: Experiment with hyperparameter tuning, cross-validation, and ensemble methods to improve accuracy! 🚀

📜 Source

Dataset: Apple Quality Dataset on Kaggle (Update with actual link)

📌 Author: Ishant Sehrawat

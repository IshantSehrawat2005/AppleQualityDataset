<!DOCTYPE html>
<html>
<head>
    <title>Decision Tree Classifier</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 900px; margin: auto; padding: 20px; }
        h1, h2 { color: #2c3e50; }
        code { background-color: #f4f4f4; padding: 3px; border-radius: 4px; }
        pre { background-color: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }
    </style>
</head>
<body>
    <h1>Decision Tree Classifier</h1>
    
    <h2>📌 Overview</h2>
    <p>This project implements a <strong>Decision Tree Classifier</strong> using Scikit-Learn to analyze and predict outcomes based on a dataset. The model undergoes data preprocessing, training, feature selection, and evaluation.</p>
    
    <h2>📂 Table of Contents</h2>
    <ul>
        <li><a href="#dataset">Dataset</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#preprocessing">Preprocessing</a></li>
        <li><a href="#model-training">Model Training</a></li>
        <li><a href="#feature-selection">Feature Selection</a></li>
        <li><a href="#evaluation">Evaluation</a></li>
        <li><a href="#results">Results</a></li>
        <li><a href="#insights">Insights & Learnings</a></li>
        <li><a href="#future">Future Enhancements</a></li>
        <li><a href="#conclusion">Conclusion</a></li>
        <li><a href="#source">Source</a></li>
    </ul>
    
    <h2 id="dataset">📊 Dataset</h2>
    <p>The dataset used is the <strong>Apple Quality Dataset</strong> from <strong>Kaggle</strong>.</p>
    <p><a href="https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality">Apple Quality Dataset on Kaggle</a></p>
    
    <h2 id="installation">⚙️ Installation</h2>
    <pre><code>pip install numpy pandas scikit-learn</code></pre>
    
    <h2 id="preprocessing">🔄 Preprocessing</h2>
    <pre><code>from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)</code></pre>
    
    <h2 id="model-training">🌳 Model Training</h2>
    <pre><code>from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=16, random_state=8)
clf.fit(X_train_scaled, y_train)</code></pre>
    
    <h2 id="feature-selection">🏆 Feature Selection</h2>
    <pre><code>importances = clf.feature_importances_
threshold = 0.1  # Adjust threshold as needed
selected_features = X.columns[importances > threshold]
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]</code></pre>
    
    <h2 id="evaluation">📈 Evaluation</h2>
    <pre><code>from sklearn.metrics import accuracy_score
y_pred = clf.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred) * 100
print(f"Accuracy Score: {acc:.2f}%")</code></pre>
    
    <h2 id="results">📊 Results</h2>
    <ul>
        <li>✅ <strong>Feature Importance Analysis</strong> helps in reducing irrelevant features.</li>
        <li>✅ <strong>Decision Tree Model</strong> achieves an accuracy of <strong>X%</strong> (update with actual result).</li>
        <li>✅ <strong>Scalability</strong>: The model can be fine-tuned using <code>max_depth</code>, <code>criterion</code>, and other hyperparameters.</li>
    </ul>
    
    <h2 id="insights">💡 Insights & Learnings</h2>
    <ul>
        <li><strong>Feature Scaling</strong> improved model performance.</li>
        <li><strong>Feature Selection</strong> optimized efficiency.</li>
        <li><strong>Max Depth Impact</strong>: Balancing between underfitting and overfitting.</li>
        <li><strong>Hyperparameter Tuning</strong> can improve accuracy.</li>
    </ul>
    
    <h2 id="future">🚀 Future Enhancements</h2>
    <ul>
        <li>Experiment with <strong>Random Forest</strong> and <strong>Gradient Boosting</strong>.</li>
        <li>Implement <strong>Hyperparameter Optimization</strong> using GridSearchCV.</li>
        <li>Use <strong>Cross-Validation</strong> for robust performance evaluation.</li>
        <li>Deploy the model as a <strong>web API</strong>.</li>
    </ul>
    
    <h2 id="conclusion">🏁 Conclusion</h2>
    <p>Decision Trees are powerful for classification tasks, and Feature Selection improves efficiency. Further optimization can be achieved through hyperparameter tuning and ensemble methods.</p>
    
    <h2 id="source">📜 Source</h2>
    <p>Dataset: <a href="https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality">Apple Quality Dataset on Kaggle</a></p>
</body>
</html>



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)

# Load data
file_path = "robotic_arm_fault_dataset_correlated.csv"
data = pd.read_csv(file_path)

# Features and target
X = data.drop(columns=['Time', 'Fault'])
y = data['Fault']

# Train-test split (stratified to preserve class ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize SGD Classifier
sgd_model = SGDClassifier(loss="log_loss", max_iter=1000, random_state=42)
sgd_model.fit(X_train, y_train)

# Predictions
y_pred = sgd_model.predict(X_test)
y_prob = sgd_model.predict_proba(X_test)[:, 1]

# Performance metrics
print("SGDClassifier Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=[0,1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - SGDClassifier")
plt.show()

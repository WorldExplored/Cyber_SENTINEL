

#Best Hyperparameters: {'C': 1, 'gamma': 0.01, 'kernel': 'linear'}

import pandas as pd
import re
from scipy.io import arff
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
import shap
import numpy as np
import warnings
import time

start_time = time.time()

# Suppress XGBoost binary model format warnings
warnings.filterwarnings("ignore", message=".*Saving into deprecated binary model format.*")

# Mapping dictionary between the short names and the descriptive names
feature_mapping = {
    '-1': 'having_IP_Address',
    '1': 'URL_Length',
    '1.1': 'Shortining_Service',
    '1.2': 'having_At_Symbol',
    '-1.1': 'double_slash_redirecting',
    '-1.2': 'Prefix_Suffix',
    '-1.3': 'having_Sub_Domain',
    '-1.4': 'SSLfinal_State',
    '-1.5': 'Domain_registeration_length',
    '1.3': 'Favicon',
    '1.4': 'port',
    '-1.6': 'HTTPS_token',
    '1.5': 'Request_URL',
    '-1.7': 'URL_of_Anchor',
    '1.6': 'Links_in_tags',
    '-1.8': 'SFH',
    '-1.9': 'Submitting_to_email',
    '-1.10': 'Abnormal_URL',
    '0': 'Redirect',
    '1.7': 'on_mouseover',
    '1.8': 'RightClick',
    '1.9': 'popUpWidnow',
    '1.10': 'Iframe',
    '-1.11': 'age_of_domain',
    '-1.12': 'DNSRecord',
    '-1.13': 'web_traffic',
    '-1.14': 'Page_Rank',
    '1.11': 'Google_Index',
    '1.12': 'Links_pointing_to_page',
    '-1.15': 'Statistical_report',
    'Result': 'Result'
}


# Read the content of the .arff file
with open('Training_Dataset.arff', 'r') as file:
    arff_content = file.read()

# Replace the short feature names with descriptive names using regular expressions
for short_name, descriptive_name in feature_mapping.items():
    arff_content = re.sub(r'\b' + re.escape(short_name) + r'\b', descriptive_name, arff_content)

# Write the updated content back to the .arff file
with open('Training_Dataset_updated.arff', 'w') as file:
    file.write(arff_content)

# Load the updated .arff file
data, meta = arff.loadarff('Training_Dataset_updated.arff')

# Get the list of attribute names from the meta data
attribute_names = meta.names()

# Split the data into features (X) and target (y)
X = pd.DataFrame(data).drop(columns=['Result'])
y = pd.DataFrame(data)['Result']

# Encode categorical variables if any
label_encoder = LabelEncoder()
X_encoded = X.apply(label_encoder.fit_transform)
y_encoded = label_encoder.fit_transform(y)

# Standardize the features
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X_encoded)

# Initialize StratifiedKFold for k-fold cross-validation
skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

# Lists to store evaluation metrics across folds
accuracies = []
precisions = []
recalls = []
f1_scores = []
roc_auc_scores = []

# Lists to store SHAP values for each fold
shap_values_list = []

for train_index, test_index in skf.split(X_standardized, y_encoded):
    X_train, X_test = X_standardized[train_index], X_standardized[test_index]
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]

    # Create an SVM classifier
    svm_classifier = SVC(probability=True, kernel='linear', C=1.0, gamma=0.01)

    # Train the classifier on the training data
    svm_classifier.fit(X_train, y_train)

    # Predict the target values on the test data
    y_pred = svm_classifier.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Calculate ROC AUC score
    y_prob = svm_classifier.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)

    # Store metrics in lists
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    roc_auc_scores.append(roc_auc)

    # SHAP interpretation
    explainer = shap.Explainer(svm_classifier, X_train)
    shap_values = explainer(X_test)
    shap_values_list.append(shap_values)

end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Print the elapsed time
print(f"Elapsed time for model execution: {elapsed_time} seconds")
# Calculate mean and standard deviation of evaluation metrics
mean_accuracy = round(sum(accuracies) / len(accuracies), 2)
mean_precision = round(sum(precisions) / len(precisions), 2)
mean_recall = round(sum(recalls) / len(recalls), 2)
mean_f1_score = round(sum(f1_scores) / len(f1_scores), 2)
mean_roc_auc_score = round(sum(roc_auc_scores) / len(roc_auc_scores), 2)

std_accuracy = round(np.std(accuracies), 2)
std_precision = round(np.std(precisions), 2)
std_recall = round(np.std(recalls), 2)
std_f1_score = round(np.std(f1_scores), 2)
std_roc_auc_score = round(np.std(roc_auc_scores), 2)

print(f"Mean Accuracy: {mean_accuracy} ± {std_accuracy}")
print(f"Mean Precision: {mean_precision} ± {std_precision}")
print(f"Mean Recall: {mean_recall} ± {std_recall}")
print(f"Mean F1-score: {mean_f1_score} ± {std_f1_score}")
print(f"Mean ROC AUC Score: {mean_roc_auc_score} ± {std_roc_auc_score}")

# Concatenate the SHAP values from all folds
concatenated_shap_values = np.concatenate([shap_values.values for shap_values in shap_values_list])

# Generate SHAP summary plot using the concatenated SHAP values
shap.summary_plot(concatenated_shap_values, X_test, plot_type="bar")

#Hyperparamter finding code: 
'''

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

# Load a dataset (e.g., the Iris dataset)
dataset = datasets.load_iris()
X = dataset.data
y = dataset.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a grid of hyperparameters to search over
param_grid = {
    'C': [0.1, 1, 10],          # Regularization parameter
    'kernel': ['linear', 'rbf'], # Kernel type (linear or radial basis function)
    'gamma': [0.01, 0.1, 1]     # Kernel coefficient for 'rbf' kernel
}

# Create an SVM classifier
svm = SVC()

# Create a GridSearchCV object with cross-validation
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')

# Perform the grid search on the training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters from the grid search
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Train an SVM model with the best hyperparameters on the entire training set
best_svm = SVC(**best_params)
best_svm.fit(X_train, y_train)

# Evaluate the best model on the test set
y_pred = best_svm.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print("Test Accuracy:", accuracy)'''
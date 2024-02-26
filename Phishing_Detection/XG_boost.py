


#SHAP
import pandas as pd
import re
from scipy.io import arff
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
#import seaborn as sns
#import matplotlib.pyplot as plt
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

best_params = {'colsample_bytree': 0.9, 'learning_rate': 0.2, 'max_depth': 5, 'n_estimators': 300, 'subsample': 0.9}

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

for train_index, test_index in skf.split(X_encoded, y_encoded):
    X_train, X_test = X_encoded.iloc[train_index], X_encoded.iloc[test_index]
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]

    # Create an XGBoost classifier
    xgb_classifier = xgb.XGBClassifier(**best_params)

    # Train the classifier on the training data
    xgb_classifier.fit(X_train, y_train)

    # Predict the target values on the test data
    y_pred = xgb_classifier.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Calculate ROC AUC score
    y_prob = xgb_classifier.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)

    # Store metrics in lists
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    roc_auc_scores.append(roc_auc)

    # SHAP interpretation
    explainer = shap.Explainer(xgb_classifier)
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

# Select the first set of SHAP values and datashap_values = shap_values_list[0].valuesdata = shap_values_list[0].data

# Generate SHAP summary plotshap.summary_plot(shap_values, data, plot_type="bar")
#K-Fold Validation Vers.
'''import pandas as pd
import re
from scipy.io import arff
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc

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

best_params = {'colsample_bytree': 0.9, 'learning_rate': 0.2, 'max_depth': 5, 'n_estimators': 300, 'subsample': 0.9}

# Read the content of the .arff file
with open('Training_Dataset.arff', 'r') as file:
    arff_content = file.read()

# Replace the short feature names with descriptive names using regular expressions
for short_name, descriptive_name in feature_mapping.items():
    arff_content = re.sub(r'\b' + re.escape(short_name) + r'\b', descriptive_name, arff_content)

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

# Initialize k-fold cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store evaluation metrics
accuracies = []
precisions = []
recalls = []
f1_scores = []
roc_auc_scores = []

# Perform k-fold cross-validation
for train_idx, test_idx in kfold.split(X_encoded, y_encoded):
    X_train, X_test = X_encoded.iloc[train_idx], X_encoded.iloc[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    # Create an XGBoost classifier with the best hyperparameters
    xgb_classifier = xgb.XGBClassifier(**best_params)

    # Train the classifier on the training data
    xgb_classifier.fit(X_train, y_train)

    # Predict the target values on the test data
    y_pred = xgb_classifier.predict(X_test)

    # Calculate and store evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    precision = precision_score(y_test, y_pred)
    precisions.append(precision)

    recall = recall_score(y_test, y_pred)
    recalls.append(recall)

    f1 = f1_score(y_test, y_pred)
    f1_scores.append(f1)

    y_prob = xgb_classifier.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    roc_auc_scores.append(roc_auc)

# Calculate the mean and standard deviation of evaluation metrics
mean_accuracy = sum(accuracies) / len(accuracies)
std_accuracy = (sum((x - mean_accuracy) ** 2 for x in accuracies) / len(accuracies)) ** 0.5

mean_precision = sum(precisions) / len(precisions)
std_precision = (sum((x - mean_precision) ** 2 for x in precisions) / len(precisions)) ** 0.5

mean_recall = sum(recalls) / len(recalls)
std_recall = (sum((x - mean_recall) ** 2 for x in recalls) / len(recalls)) ** 0.5

mean_f1_score = sum(f1_scores) / len(f1_scores)
std_f1_score = (sum((x - mean_f1_score) ** 2 for x in f1_scores) / len(f1_scores)) ** 0.5

mean_roc_auc_score = sum(roc_auc_scores) / len(roc_auc_scores)
std_roc_auc_score = (sum((x - mean_roc_auc_score) ** 2 for x in roc_auc_scores) / len(roc_auc_scores)) ** 0.5

# Print the mean and standard deviation of evaluation metrics
print(f'Mean Accuracy: {mean_accuracy:.2f} ± {std_accuracy:.2f}')
print(f'Mean Precision: {mean_precision:.2f} ± {std_precision:.2f}')
print(f'Mean Recall: {mean_recall:.2f} ± {std_recall:.2f}')
print(f'Mean F1-score: {mean_f1_score:.2f} ± {std_f1_score:.2f}')
print(f'Mean ROC AUC Score: {mean_roc_auc_score:.2f} ± {std_roc_auc_score:.2f}')'''




#WORKING CODE + Hyperparameters, HAS GRAPHS FOR IMPORTANT FEAUTES, AS WELL AS ROC AURC Curve if needed.
"""import pandas as pd
import re
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV

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

best_params = {'colsample_bytree': 0.9, 'learning_rate': 0.2, 'max_depth': 5, 'n_estimators': 300, 'subsample': 0.9}

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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

# Create an XGBoost classifier
xgb_classifier = xgb.XGBClassifier()
xgb_classifier = xgb.XGBClassifier(**best_params)

# Train the classifier on the training data
xgb_classifier.fit(X_train, y_train)

# Predict the target values on the test data
y_pred = xgb_classifier.predict(X_test)

feature_importance = xgb_classifier.feature_importances_

# Create a DataFrame to store feature names and their corresponding importances
feature_importance_df = pd.DataFrame({'Feature': X_encoded.columns, 'Importance': feature_importance})

# Sort features by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Generate a classification report
report = classification_report(y_test, y_pred)
print('Classification Report:\n', report)

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:\n', cm)

# Calculate precision, recall, and F1-score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-score: {f1:.2f}')

# Calculate ROC AUC score
y_prob = xgb_classifier.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)

print(f'ROC AUC Score: {roc_auc:.2f}')

# Generate ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)


plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, orient='h')
plt.title('Feature Importance Plot')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()"""




#Finding Best Hyperparameter code:
'''import pandas as pd
import re
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import GridSearchCV

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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],        # Number of trees in the forest
    'max_depth': [3, 4, 5],                # Maximum depth of the tree
    'learning_rate': [0.01, 0.1, 0.2],    # Learning rate
    'subsample': [0.8, 0.9, 1.0],         # Fraction of samples used for fitting the trees
    'colsample_bytree': [0.8, 0.9, 1.0],  # Fraction of features used for fitting the trees
}

# Create an XGBoost classifier
xgb_classifier = xgb.XGBClassifier()

# Create GridSearchCV
grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, 
                           scoring='roc_auc', cv=3, verbose=2, n_jobs=-1)

# Perform the grid search
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best Hyperparameters:")
print(grid_search.best_params_)

# Get the best model
best_xgb_classifier = grid_search.best_estimator_

# Predict the target values on the test data using the best model
y_pred = best_xgb_classifier.predict(X_test)

# Calculate ROC AUC score with the best model
y_prob = best_xgb_classifier.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)

print(f'Best ROC AUC Score: {roc_auc:.2f}')'''
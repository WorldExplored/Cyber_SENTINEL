'''
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
import time
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

sys.path.append("/Users/srreyanshsethi/anaconda3/envs/intrusion/lib/python3.10/site-packages")

start_time = time.time()
data = pd.read_csv('combined_data.csv', low_memory=False)

data = data.sample(n=1000, random_state=42)
data = data.astype(str)

# Define features and target variable
X = data.drop(' Label', axis=1)
y = data[' Label']

problematic_columns = [0, 1, 3, 6]
print(X.columns[problematic_columns])

X.drop(X.columns[problematic_columns], axis=1, inplace=True)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the CatBoost classifier with early stopping
model = CatBoostClassifier(iterations=200,
                           depth=8,
                           learning_rate=0.15,
                           loss_function='MultiClass',
                           cat_features=list(range(X.shape[1])),
                           verbose=100,
                           thread_count=4,
                           early_stopping_rounds=10,  # Number of rounds to wait for improvement
                           eval_metric='MultiClass',  # Metric to use for early stopping
                           )

# Fit the model on the training set and validate on the validation set
model.fit(X_train, y_train, eval_set=(X_val, y_val))

# Predict on the full dataset
y_pred = model.predict(X)

# Convert numerical labels to string to ensure compatibility in crosstab
y = y.astype(str)
y_pred = y_pred.astype(str)

# Ensure y and y_pred have the same length
num_samples = min(len(y), len(y_pred))
y = y[:num_samples]
y_pred = y_pred[:num_samples]

# Calculate confusion matrix
conf_matrix = confusion_matrix(y, y_pred)

# Plot the confusion matrix using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Evaluate the model using precision, recall, and F1-score
report = classification_report(y, y_pred)
print(report)
 #^^Working Early stopping + preproccessing! '''
#Preproccessing:
"""import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
import shap
import time
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

sys.path.append("/Users/srreyanshsethi/anaconda3/envs/intrusion/lib/python3.10/site-packages")

start_time = time.time()
data = pd.read_csv('combined_data.csv', low_memory=False)
data = data.sample(n=1000, random_state=42)
data = data.astype(str)

 
# Define features and target variable
X = data.drop(' Label', axis=1)
y = data[' Label']

problematic_columns = [0, 1, 3, 6]
X.drop(X.columns[problematic_columns], axis=1, inplace=True)


    # Initialize the CatBoost classifier
model = CatBoostClassifier(iterations=200,
                        depth=8,
                        learning_rate=0.15,
                        loss_function='MultiClass',
                        cat_features=list(range(X.shape[1])),
                        verbose=100,
                        thread_count=4)

    # Fit the model on the entire dataset
model.fit(X, y)

y_pred = model.predict(X)

# Convert numerical labels to string to ensure compatibility in crosstab
y = y.astype(str)
y_pred = y_pred.astype(str)

# Ensure y and y_pred have the same length
num_samples = min(len(y), len(y_pred))
y = y[:num_samples]
y_pred = y_pred[:num_samples]

# Calculate confusion matrix
conf_matrix = confusion_matrix(y, y_pred)

# Plot the confusion matrix using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()"""

# Explore misclassified examples
#misclassified_indices = y.index[y != y_pred]
#misclassified_samples = data.loc[misclassified_indices]

# Display some misclassified samples
#print("\nMisclassified Samples:")
#print(misclassified_samples)


    # Evaluate the model using precision, recall, and F1-score
'''report = classification_report(y, y_pred)
print(report)'''

    # Explain the model using SHAP
'''explainer = shap.Explainer(model)
shap_values = explainer(X)

end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

intar = X.to_numpy(dtype=int)

# Print the elapsed time
print(f"Elapsed time for model execution: {elapsed_time} seconds")

# Visualize SHAP values (e.g., summary plot)
shap.summary_plot(shap_values, intar, feature_names=list(X.columns))'''


#K-FOLD UPDATE CODE Vers. 2
'''
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer

data = pd.read_csv('combined_data.csv', low_memory=False)

data = data.sample(n=1000, random_state=42)

data = data.astype(str)

# Define features and target variable
X = data.drop(' Label', axis=1)
y = data[' Label']

problematic_columns = [0, 1, 3, 6, 84]
data.drop(data.columns[problematic_columns], axis=1, inplace=True)

############ FEATURE ENGINEERING ############
# Add feature: Word Count
X['Word_Count'] = X.apply(lambda row: len(row.str.split()), axis=1)

# Add feature: Presence of 'important_keyword'
important_keyword = 'important_keyword'  # Replace with your keyword
X['Has_Important_Keyword'] = X.apply(lambda row: 1 if important_keyword in row.values else 0, axis=1)

# Add feature: TF-IDF of the text
tfidf_vectorizer = CountVectorizer()
tfidf_features = tfidf_vectorizer.fit_transform(X.iloc[:, 0])  # Assuming the text is in the first column
X['TF-IDF'] = tfidf_features.toarray().tolist()

# Convert 'TF-IDF' feature to a string representation
X['TF-IDF'] = X['TF-IDF'].apply(lambda x: ' '.join(map(str, x)))

#############

# Initialize the StratifiedKFold cross-validator
num_folds = 5  # You can change this to your desired number of folds
kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

# Choose an appropriate loss function for multiclass classification
loss_function = 'MultiClass'

# Initialize a list to store classification reports for each fold
classification_reports = []

# Perform K-fold cross-validation
for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
    print(f"Fold {fold+1}/{num_folds}")

    # Split the data into training and testing sets for this fold
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Initialize the CatBoost classifier
    model = CatBoostClassifier(iterations=200,
                               depth=8,
                               learning_rate=0.15,
                               loss_function=loss_function,
                               cat_features=list(range(X_train.shape[1])),
                               verbose=100,
                               thread_count=4)

    # Fit the model on the training data for this fold
    model.fit(X_train, y_train)

    # Make predictions on the test set for this fold
    y_pred = model.predict(X_test)

    # Evaluate the model using precision, recall, and F1-score for this fold
    report = classification_report(y_test, y_pred)
    print(report)

    # Append the classification report to the list
    classification_reports.append(report)

# Calculate and print the average classification report across all folds
avg_classification_report = "\nAverage Classification Report Across All Folds:\n"
for metric in classification_reports[0].split('\n')[:3]:
    avg_metric_values = [
        sum(float(report.split('\n')[i].split()[-1]) for report in classification_reports) / num_folds
        for i in range(3)
    ]
    avg_classification_report += f"{metric} {avg_metric_values[0]:.2f} {avg_metric_values[1]:.2f} {avg_metric_values[2]:.2f}\n"

print(avg_classification_report)
'''

#OLD CODE vers. 1
'''import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer

data = pd.read_csv('combined_data.csv', low_memory=False)

data = data.sample(n=1000, random_state=42)

data = data.astype(str)

# Define features and target variable
X = data.drop(' Label', axis=1)
y = data[' Label']

problematic_columns = [0, 1, 3, 6, 84]
data.drop(data.columns[problematic_columns], axis=1, inplace=True)


############FEATURE ENGINEERING 
# Add feature: Word Count
X['Word_Count'] = X.apply(lambda row: len(row.str.split()), axis=1)

# Add feature: Presence of 'important_keyword'
important_keyword = 'important_keyword'  # Replace with your keyword
X['Has_Important_Keyword'] = X.apply(lambda row: 1 if important_keyword in row.values else 0, axis=1)

# Add feature: TF-IDF of the text
tfidf_vectorizer = CountVectorizer()
tfidf_features = tfidf_vectorizer.fit_transform(X.iloc[:, 0])  # Assuming the text is in the first column
X['TF-IDF'] = tfidf_features.toarray().tolist()

# Convert 'TF-IDF' feature to a string representation
X['TF-IDF'] = X['TF-IDF'].apply(lambda x: ' '.join(map(str, x)))

#############





# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose an appropriate loss function for multiclass classification
loss_function = 'MultiClass'

# Initialize the CatBoost classifier
model = CatBoostClassifier(iterations=200,
                           depth=8,
                           learning_rate=0.15,
                           loss_function=loss_function,
                           cat_features=list(range(X_train.shape[1])),
                           verbose=100,
                           thread_count=4)

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using precision, recall, and F1-score
report = classification_report(y_test, y_pred)
print(report)
'''


import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Load the data
data = pd.read_csv('combined_data.csv', low_memory=False)

data = data.astype(str)

# Define features and target variable
X = data.drop(' Label', axis=1)
y = data[' Label']


problematic_columns = [
    'Flow ID',
    ' Fwd Header Length.1',
    ' Destination Port',
    ' Source IP',
    ' Source Port',
    ' Destination IP',
    ' Timestamp',
    'Init_Win_bytes_forward',
    ' Init_Win_bytes_backward',
    'Idle Mean',
    ' Idle Std',
    ' Idle Max',
    ' Idle Min',
    'Active Mean',
    ' Active Std',
    ' Active Max',
    ' Active Min',
    ' SYN Flag Count',
    ' RST Flag Count',
    ' PSH Flag Count',
    ' ACK Flag Count',
    ' URG Flag Count',
    ' CWE Flag Count',
    ' ECE Flag Count',
    'Fwd Avg Bytes/Bulk',
    ' Fwd Avg Packets/Bulk',
    ' Fwd Avg Bulk Rate',
    ' Bwd Avg Bytes/Bulk',
    ' Bwd Avg Packets/Bulk',
    'Bwd Avg Bulk Rate'
]







X = X.drop(columns=problematic_columns)

# Split the data into training and validation sets using stratified sampling
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize CatBoostClassifier
model = CatBoostClassifier(
    iterations=200,
    depth=8,
    learning_rate=0.15,
    loss_function='MultiClass',
    cat_features=list(range(X_train.shape[1])),
    verbose=100,
    thread_count=4,
    early_stopping_rounds=10,
    eval_metric='MultiClass'
)

# Fit the model
start_time = time.time()
model.fit(X_train, y_train, eval_set=(X_val, y_val))

# Predict on the validation set
y_pred = model.predict(X_val)

# Calculate confusion matrix and classification report
conf_matrix = confusion_matrix(y_val, y_pred)
report = classification_report(y_val, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Print classification report
print(report)

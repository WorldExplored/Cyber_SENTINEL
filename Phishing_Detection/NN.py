#Best parameters found:  {'activation': 'relu', 'dropout_rate': 0.2, 'hidden_layers': 2, 'neurons': 64, 'optimizer': 'adam'}
import pandas as pd
import re
from scipy.io import arff
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import shap
import numpy as np
import warnings
import time

start_time = time.time()
# Suppress XGBoost binary model format warnings
warnings.filterwarnings("ignore", message=".*Saving into deprecated binary model format.*")

# Mapping dictionary between the short names and the descriptive names
# ... (rest of your code)

# Load the updated .arff file
data, meta = arff.loadarff('Training_Dataset_updated.arff')

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

    # Create a feedforward neural network model
    model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dropout(0.2),  # Dropout rate based on best hyperparameters
    keras.layers.Dense(64, activation='relu'),  # Increased number of neurons based on best hyperparameters
    keras.layers.Dropout(0.2),  # Dropout rate based on best hyperparameters
    keras.layers.Dense(1, activation='sigmoid')
    ])


    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model on the training data
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    # Evaluate the model on the test data
    y_prob = model.predict(X_test)
    y_pred = (y_prob > 0.5).astype(int)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    # Store metrics in lists
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    roc_auc_scores.append(roc_auc)

    # SHAP interpretation
    explainer = shap.Explainer(model, X_train)
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


#Hyperparameter finding code: 
'''import pandas as pd
import re
from scipy.io import arff
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import make_scorer, roc_auc_score
import shap
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.base import BaseEstimator, ClassifierMixin

# Mapping dictionary between the short names and the descriptive names
# ... (rest of your code)

# Load the updated .arff file
data, meta = arff.loadarff('Training_Dataset_updated.arff')

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

# Function to create a neural network model
def create_nn(optimizer='adam', activation='relu', neurons=64, hidden_layers=1, dropout_rate=0.2):
    model = Sequential()
    model.add(Dense(neurons, input_dim=X_standardized.shape[1], activation=activation))
    
    for _ in range(hidden_layers):
        model.add(Dense(neurons, activation=activation))
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Create a custom classifier class that wraps the neural network
class NNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, optimizer='adam', activation='relu', neurons=64, hidden_layers=1, dropout_rate=0.2):
        self.optimizer = optimizer
        self.activation = activation
        self.neurons = neurons
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.model = None
    
    def fit(self, X, y):
        self.model = create_nn(self.optimizer, self.activation, self.neurons, self.hidden_layers, self.dropout_rate)
        self.model.fit(X, y, epochs=10, batch_size=32, verbose=0)
        self.classes_ = np.unique(y)  # Define the classes_ attribute
        
    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype(int)


# Define hyperparameters to search
param_grid = {
    'optimizer': ['adam', 'rmsprop'],
    'activation': ['relu', 'tanh'],
    'neurons': [32, 64],
    'hidden_layers': [1, 2],
    'dropout_rate': [0.2, 0.3]
}

# Create GridSearchCV
grid = GridSearchCV(estimator=NNClassifier(), param_grid=param_grid, scoring=make_scorer(roc_auc_score), cv=5, n_jobs=-1)

# Fit the grid search to your data
grid_result = grid.fit(X_standardized, y_encoded)

# Print the best parameters and ROC AUC score
print("Best parameters found: ", grid_result.best_params_)
print("Best ROC AUC score: {:.4f}".format(grid_result.best_score_))

# You can access the best model using grid_result.best_estimator_
best_model = grid_result.best_estimator_'''
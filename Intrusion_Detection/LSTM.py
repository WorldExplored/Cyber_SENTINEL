'''

#Best Hyperparameters:  {'lstm_units': 50, 'dropout_rate': 0.2, 'epochs': 10}

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt


# Load the data
data = pd.read_csv('combined_data.csv', dtype=str)

# Drop irrelevant columns
drop_columns = ['Flow ID', ' Source IP', ' Source Port', ' Destination IP', ' Destination Port']
data.drop(drop_columns, axis=1, inplace=True)

# Convert 'Timestamp' to datetime and extract numeric representation
data[' Timestamp'] = pd.to_datetime(data[' Timestamp'], errors='coerce').values.astype(np.int64)

# Identify numeric columns
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

# Convert numeric columns to float, handling non-numeric values
for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Calculate mean only for numeric columns
numeric_mean = data[numeric_cols].mean()

# Replace NaNs in numeric columns with their means
data[numeric_cols] = data[numeric_cols].fillna(numeric_mean)

# Separate target variable
y = data[' Label']
data.drop([' Label'], axis=1, inplace=True)

# Scale numeric features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[numeric_cols])

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(data_scaled, y_encoded, test_size=0.3, random_state=42)

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Reshape input for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')
'''


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import keras_tuner as kt
from keras_tuner import RandomSearch

data = pd.read_csv('combined_data.csv')

# Remove leading and trailing spaces from column names
data.columns = data.columns.str.strip()

# Drop irrelevant columns, ensuring column names match exactly
drop_columns = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Timestamp']
data.drop(columns=drop_columns, errors='ignore', inplace=True)

# Convert all columns to numeric, coercing errors into NaNs
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Now identify numeric columns
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

# Replace NaNs in numeric columns with their means
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Separate target variable
y = data['Label']
data.drop(columns=['Label'], errors='ignore', inplace=True)


# Scale numeric features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[numeric_cols])

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(data_scaled, y_encoded, test_size=0.3, random_state=42)

# Reshape input for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

def build_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32),
                   return_sequences=True,
                   input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32)))
    model.add(Dropout(hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(
        optimizer=Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Initialize the tuner
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,  # Set the number of experiments to conduct
    executions_per_trial=1,
    directory='tuner_results',
    project_name='lstm_hyperparameter_tuning'
)

# Perform hyperparameter tuning
tuner.search(X_train, y_train, epochs=10, validation_split=0.2, verbose=2)

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The best number of units in the first LSTM layer is {best_hps.get('units')} and
the best dropout rate in the first LSTM layer is {best_hps.get('dropout_1')}.
The best number of units in the second LSTM layer is {best_hps.get('units')} and
the best dropout rate in the second LSTM layer is {best_hps.get('dropout_2')}.
The best learning rate for the optimizer is {best_hps.get('learning_rate')}.
""")

# Build the model with the optimal hyperparameters and train it on the data
model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=2)

# Evaluate the tuned model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')
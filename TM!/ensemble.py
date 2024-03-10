from keras.models import load_model
import joblib
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd


#Averages for each column in each dataset incase user doesn't have input
intrusion_average = [5.0156875, 4.970425, 732.2017875, 6633.8004, 384.3607875, 26.3388375, 123.089608396005, 148.97975071357772, 2855.7985625, 18.6761, 934.1703422642775, 1278.1327970477223, 725511.7349966656, 16093.931847482485, 1441983.7839274046, 3829415.6153083383, 12067771.7080875, 15599.5347875, 14261085.6728625, 2328306.0471473625, 4610956.780403548, 11582174.1960375, 227847.1053125, 6537182.35005, 956646.6753867052, 1488924.0903169059, 4261550.8161125, 265928.7737375, 0.03365, 0.0, 0.0, 0.0, 116.4869, 118.30265, 14370.00789016537, 1720.9063451156787, 8.6672125, 3185.7674375, 515.3643658249833, 1073.587306891672, 2739508.842652506, 0.003125, 1.015075, 574.8532287704479, 123.089608396005, 934.17034226429, 5.0156875, 732.2017875, 4.970425, 6633.8004, 3.36115, 21.82755]

malware_average = ['nan' , 0.51031, 1487796144.1214, 10.34509, 0.6439433333333333, 0.009756666666666667, 0.43827333333333335, 0.46056, 1.8668466666666668, 0.6574733333333334, 0.0071133333333333335, 1.11585, 0.33683, 1.5291733333333333, 0.45637666666666665]

phishing_average = [0.6568973315241972, 0.36680235187697874, 0.8693803708729082, 0.8502939846223428, 0.870737222976029, 0.13251922207146088, 1.0639529624604251, 1.250927182270466, 0.3316146540027137, 0.8142921754862054, 0.8641338760741746, 0.8375395748530077, 0.5933966530981456, 0.9234735413839892, 0.8818634102216192, 0.40425146992311173, 0.8178199909543193, 0.8526458616010855, 0.11569425599276345, 0.8810492989597467, 0.9569425599276346, 0.8066938037087291, 0.9084577114427861, 0.5306196291270918, 0.6885572139303483, 1.287290818634102, 0.25816372682044325, 0.86078697421981, 1.34400723654455, 0.8597919493441881]


# Preloaded models
lstm_intrusion_model = load_model('lstm_intrusion_detection.h5')
catboost_intrusion_model = joblib.load('catboost_intrusion_detection.pkl')
alohadl_malware_model = torch.load('alohadl_malware_detection.pth')
xgboost_malware_model = joblib.load('xgboost_malware_detection.pkl')
lightgmb_malware_model = joblib.load('lightgmb_malware_detection.pkl')
nn_phishing_model = load_model('nn_phishing_detection.h5')
svm_phishing_model = joblib.load('svm_phishing_detection.pkl')
xgboost_phishing_model = joblib.load('xgboost_phishing_detection.pkl')


scaler_intrusion = StandardScaler()
scaler_alohadl = StandardScaler()
scaler_xgboost = StandardScaler()
scaler_svm = StandardScaler()
scaler_nn = StandardScaler()
scaler_xgboost_phishing = StandardScaler()

# Placeholder for input dimensions for models expecting specific input shapes
input_dim_alohadl = 100 

# Assuming model-specific preprocessing functions are defined as:
# preprocess_for_intrusion, preprocess_for_malware, preprocess_for_phishing
# These should align with your model training preprocessing

def ensemble_predict(models, input_data, domain):
    votes = []
    for model in models:
        if domain == "intrusion":
            # Example: LSTM needs reshaped data
            if "lstm" in str(type(model)):
                input_data_reshaped = input_data.reshape((input_data.shape[0], 1, -1))
                prediction = model.predict(input_data_reshaped)
            else:
                prediction = model.predict(input_data)
        elif domain == "malware":
            # Example: ALOHA model needs PyTorch tensor
            if "ALOHA" in str(model.__class__):
                input_data_tensor = torch.FloatTensor(input_data)
                prediction = model(input_data_tensor).numpy()
            else:
                prediction = model.predict(input_data)
        elif domain == "phishing":
            # Direct prediction for models like SVM, NN
            prediction = model.predict(input_data)
        else:
            raise ValueError("Unknown domain")

        # Simplify by considering prediction directly for binary classification
        # Adjust this based on how your models output predictions
        vote = np.round(prediction).astype(int)
        votes.extend(vote)

    # Majority voting
    final_prediction = np.round(np.mean(votes)).astype(int)
    return final_prediction

def aggregate_predictions(*predictions):
    # Assuming predictions are numpy arrays of the same length
    combined_predictions = np.vstack(predictions)
    # Perform majority voting
    final_prediction = np.round(np.mean(combined_predictions, axis=0)).astype(int)
    return final_prediction

def get_user_input(domain):
    print(f"Enter details for {domain} prediction:")
    user_input = {}

    # Define average values for each domain
    intrusion_averages = intrusion_average  # Your list of averages
    malware_averages = malware_average  # Your list of averages
    phishing_averages = phishing_average  # Your list of averages

    if domain == "phishing":
        features_required = [
            'having_IP_Address', 'URL_Length', 'Shortining_Service', 'having_At_Symbol',
            'double_slash_redirecting', 'Prefix_Suffix', 'having_Sub_Domain', 'SSLfinal_State',
            'Domain_registeration_length', 'HTTPS_token', 'Request_URL', 'URL_of_Anchor',
            'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL', 'Redirect',
            'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe', 'age_of_domain',
            'DNSRecord', 'web_traffic', 'Page_Rank', 'Google_Index', 'Links_pointing_to_page',
            'Statistical_report'
        ]
        averages = phishing_averages

    elif domain == "malware":
        features_required = [
            'file_size',  # Numerical
            'num_api_calls',  # Numerical
            'uses_encryption',  # Categorical: Yes or No
            'uses_network',  # Categorical: Yes or No
            'registry_modifications',  # Numerical
            'num_unique_ips_contacted'  # Numerical
        ]
        averages = malware_averages

    elif domain == "intrusion":
        features_required = [
            'total_fwd_packets',  # Numerical
            'total_backward_packets',  # Numerical
            'total_length_of_fwd_packets',  # Numerical
            'total_length_of_bwd_packets',  # Numerical
            'fwd_packet_length_max',  # Numerical
            'fwd_packet_length_min',  # Numerical
            'fwd_packet_length_mean',  # Numerical
            'flow_bytes_per_s',  # Numerical
            'flow_packets_per_s',  # Numerical
            'flow_iat_mean',  # Numerical
            'fwd_iat_total',  # Numerical
            'fwd_psh_flags',  # Categorical: Yes or No
            'bwd_psh_flags',  # Categorical: Yes or No
            'fwd_urg_flags',  # Categorical: Yes or No
            'bwd_urg_flags',  # Categorical: Yes or No
            'protocol',  # Categorical: e.g., TCP, UDP
        ]
        averages = intrusion_averages

    for index, feature in enumerate(features_required):
        value = input(f"{feature} (average={averages[index]}, press enter to use average): ")
        if value == "":
            user_input[feature] = averages[index]
        else:
            try:
                user_input[feature] = float(value)
            except ValueError:
                print(f"Invalid input for {feature}, using average value.")
                user_input[feature] = averages[index]

    return user_input

def main():
    domain = input("Enter the domain (intrusion/malware/phishing): ")
    user_input = get_user_input(domain)  # Adjusted to pass domain to function

    if domain == "intrusion":
        # Assuming separate preprocessing for LSTM and CatBoost as an example
        preprocessed_input_lstm = preprocess_for_intrusion_lstm(user_input, scaler_intrusion)
        preprocessed_input_catboost = preprocess_for_intrusion_catboost(user_input)
        predictions_lstm = lstm_intrusion_model.predict(preprocessed_input_lstm)
        predictions_catboost = catboost_intrusion_model.predict(preprocessed_input_catboost)
        prediction = aggregate_predictions(predictions_lstm, predictions_catboost)  # Define aggregation logic

    elif domain == "malware":
        preprocessed_input_alohadl = preprocess_for_alohadl(user_input, scaler_alohadl, input_dim_alohadl)
        preprocessed_input_lightgbm = preprocess_for_lightgbm(user_input, features_lightgbm)
        preprocessed_input_xgboost = preprocess_for_xgboost(user_input, scaler_xgboost, features_xgboost)
        predictions_alohadl = alohadl_malware_model(preprocessed_input_alohadl)  # Adjusted for torch model
        predictions_lightgbm = lightgmb_malware_model.predict(preprocessed_input_lightgbm)
        predictions_xgboost = xgboost_malware_model.predict(preprocessed_input_xgboost)
        prediction = aggregate_predictions(predictions_alohadl, predictions_lightgbm, predictions_xgboost)  # Define aggregation logic

    elif domain == "phishing":
        preprocessed_input_svm = preprocess_for_svm(user_input, scaler_svm, label_encoder_svm, features_svm)
        preprocessed_input_nn = preprocess_for_nn(user_input, scaler_nn, features_nn)
        preprocessed_input_xgboost_phishing = preprocess_for_xgboost_phishing(user_input, scaler_xgboost_phishing, features_xgboost_phishing)
        predictions_svm = svm_phishing_model.predict(preprocessed_input_svm)
        predictions_nn = nn_phishing_model.predict(preprocessed_input_nn)
        predictions_xgboost_phishing = xgboost_phishing_model.predict(preprocessed_input_xgboost_phishing)
        prediction = aggregate_predictions(predictions_svm, predictions_nn, predictions_xgboost_phishing)  # Define aggregation logic

    else:
        raise ValueError("Unknown domain")

    print(f"The prediction for {domain} is: {prediction}")

if __name__ == "__main__":
    main()

scaler_intrusion = StandardScaler()

def preprocess_for_intrusion_catboost(user_input):
    # List of columns to drop, identified from your model code
    drop_columns = ['Flow ID', ' Source IP', ' Source Port', ' Destination IP', ' Destination Port', ' Timestamp']
    # Process user input
    processed_input = {k: v for k, v in user_input.items() if k not in drop_columns}
    
    # Assuming you have a mechanism to handle missing values or all values are required in user input
    # Convert to DataFrame for CatBoost
    processed_df = pd.DataFrame([processed_input])
    
    return processed_df

def preprocess_for_intrusion_lstm(user_input, scaler_intrusion):
    # Convert user input to DataFrame
    input_df = pd.DataFrame([user_input])
    # Drop unwanted columns and fill missing values with zeros or appropriate values
    input_df = input_df.fillna(0)
    # Scale features
    input_scaled = scaler_intrusion.transform(input_df)
    # Reshape for LSTM (assuming LSTM expects 3D input)
    input_scaled_reshaped = input_scaled.reshape((1, input_scaled.shape[1], 1))
    
    return input_scaled_reshaped

def preprocess_for_lightgbm(user_input, features):
    # Remove features not used by the model
    processed_input = {k: v for k, v in user_input.items() if k in features}

    # Fill missing values
    for feature in features:
        if feature not in processed_input:
            processed_input[feature] = 0  # Default value or a value determined from your dataset

    # Convert to DataFrame for compatibility with LightGBM
    processed_df = pd.DataFrame([processed_input])
    
    return processed_df

def preprocess_for_lightgbm(user_input, features):
    # Remove features not used by the model
    processed_input = {k: v for k, v in user_input.items() if k in features}

    # Fill missing values
    for feature in features:
        if feature not in processed_input:
            processed_input[feature] = 0  # Default value or a value determined from your dataset

    # Convert to DataFrame for compatibility with LightGBM
    processed_df = pd.DataFrame([processed_input])
    
    return processed_df

def preprocess_for_alohadl(user_input, scaler, input_dim):
    # Ensure input matches expected dimension
    processed_input = np.zeros(input_dim)
    
    for i, key in enumerate(sorted(user_input.keys())):
        processed_input[i] = user_input[key]
    
    # Scale inputs
    processed_scaled = scaler.transform([processed_input])
    
    # Convert to torch tensor
    processed_tensor = torch.tensor(processed_scaled, dtype=torch.float32)
    
    return processed_tensor

def preprocess_for_xgboost(user_input, scaler, features):
    # Convert input to match expected features
    processed_input = {k: v for k, v in user_input.items() if k in features}

    # Handle missing values
    for feature in features:
        if feature not in processed_input:
            processed_input[feature] = 0  # Default or derived value

    # Convert to DataFrame for consistency
    processed_df = pd.DataFrame([processed_input])
    
    # Scale features
    processed_scaled = scaler.transform(processed_df)
    
    return processed_scaled

def preprocess_for_svm(user_input, scaler, label_encoder, features):
    # Convert input into DataFrame
    processed_input = {k: v for k, v in user_input.items() if k in features}
    
    # Encode categorical variables
    processed_encoded = label_encoder.transform([processed_input[feature] for feature in sorted(features)])
    
    # Scale features
    processed_scaled = scaler.transform([processed_encoded])
    
    return processed_scaled

def preprocess_for_nn(user_input, scaler, features):
    # Ensure input matches expected dimension
    processed_input = {k: user_input[k] for k in features if k in user_input}
    
    # Handle missing values
    for feature in features:
        if feature not in processed_input:
            processed_input[feature] = 0  # Default or derived value
    
    # Convert to DataFrame for scaling
    processed_df = pd.DataFrame([processed_input])
    
    # Scale features
    processed_scaled = scaler.transform(processed_df)
    
    return processed_scaled

def preprocess_for_xgboost_phishing(user_input, scaler, features):
    # Align user input with expected model features
    processed_input = {k: v for k, v in user_input.items() if k in features}
    
    # Fill in missing values
    for feature in features:
        if feature not in processed_input:
            processed_input[feature] = 0  # Use an appropriate default value
    
    # Convert to DataFrame for scaling
    processed_df = pd.DataFrame([processed_input])
    
    # Apply scaling
    processed_scaled = scaler.transform(processed_df)
    
    return processed_scaled

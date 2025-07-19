import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


data = pd.read_csv('/Users/satheeswaranharikrishnan/Desktop/PestPrediction/cropdata.csv')
data.columns = data.columns.str.strip()


encoder_pest = LabelEncoder()
data['Pest/Disease'] = encoder_pest.fit_transform(data['Pest/Disease'])

encoder_location = LabelEncoder()
data['Location'] = encoder_location.fit_transform(data['Location'])


features = data[['Observation Year', 'Standard Week', 'RH1(%)', 'RH2(%)', 'RF(mm)', 'WS(kmph)', 'SSH(hrs)', 'EVP(mm)', 'Location']]
target = data[['MaxT(°C)', 'MinT(°C)']]


scaler_features = MinMaxScaler()
features_scaled = scaler_features.fit_transform(features)

scaler_targets = MinMaxScaler()
target_scaled = scaler_targets.fit_transform(target)


def create_sequences(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 10
X, y = create_sequences(features_scaled, target_scaled, time_steps)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25, activation='relu'))
model.add(Dense(2)) 

model.compile(optimizer='adam', loss='mean_squared_error')


y_pred = model.predict(X_test)
y_pred_rescaled = scaler_targets.inverse_transform(y_pred)
y_test_rescaled = scaler_targets.inverse_transform(y_test)


data_predicted = data.copy()
data_predicted['Pred_MaxT'] = np.nan
data_predicted['Pred_MinT'] = np.nan


data_predicted.iloc[-len(y_pred_rescaled):, data_predicted.columns.get_loc('Pred_MaxT')] = y_pred_rescaled[:, 0]
data_predicted.iloc[-len(y_pred_rescaled):, data_predicted.columns.get_loc('Pred_MinT')] = y_pred_rescaled[:, 1]


features_pest = data_predicted[['Observation Year', 'Standard Week', 'Pred_MaxT', 'Pred_MinT', 'RH1(%)', 'RH2(%)', 'RF(mm)', 'WS(kmph)', 'SSH(hrs)', 'EVP(mm)', 'Location']]
target_pest = data_predicted['Pest/Disease']


features_pest_scaled = scaler_features.fit_transform(features_pest)


X_pest, y_pest = create_sequences(features_pest_scaled, target_pest.values, time_steps)

X_pest_train, X_pest_test, y_pest_train, y_pest_test = train_test_split(X_pest, y_pest, test_size=0.2, random_state=42)


model_pest = Sequential()
model_pest.add(LSTM(64, input_shape=(X_pest_train.shape[1], X_pest_train.shape[2]), return_sequences=True))
model_pest.add(LSTM(64, return_sequences=False))
model_pest.add(Dense(25, activation='relu'))
model_pest.add(Dense(1))  

model_pest.compile(optimizer='adam', loss='mean_squared_error')


def predict_based_on_location(location):
    
    location_encoded = encoder_location.transform([location])[0]

    
    sample_features = np.mean(features_scaled, axis=0)  
    sample_features[-1] = location_encoded 
    
    
    sequence_input_temp = np.tile(sample_features, (time_steps, 1))  
    sequence_input_temp = np.expand_dims(sequence_input_temp, axis=0)  

    
    temp_pred_scaled = model.predict(sequence_input_temp)[0] 
    temp_pred_rescaled = scaler_targets.inverse_transform([temp_pred_scaled])[0] 
    predicted_maxt = temp_pred_rescaled[0]
    predicted_mint = temp_pred_rescaled[1]

    print(f"Predicted MaxT for {location}: {predicted_maxt}")
    print(f"Predicted MinT for {location}: {predicted_mint}")

    #
    sample_features_pest = np.append(sample_features, [predicted_maxt, predicted_mint]) 
    sequence_input_pest = np.tile(sample_features_pest, (time_steps, 1))  
    sequence_input_pest = np.expand_dims(sequence_input_pest, axis=0)

    
    pest_pred_scaled = model_pest.predict(sequence_input_pest)[0]
    pest_pred_rounded = np.round(pest_pred_scaled).astype(int)
    pest_pred_label = encoder_pest.inverse_transform(pest_pred_rounded.flatten())[0] 

    print(f"Predicted Pest/Disease for {location}: {pest_pred_label}")

    return predicted_maxt, predicted_mint, pest_pred_label

# Example usage
location_input = input("Enter location: ")
predict_based_on_location(location_input)

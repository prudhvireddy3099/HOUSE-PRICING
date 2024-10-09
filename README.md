# HOUSE-PRICING
PREDICTION OF HOUSE PRICING USING MACHINE LEARING AND PYTHIN



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['age'] = 2024 - df['year_built']
    df = pd.get_dummies(df, columns=['neighborhood', 'house_style'], drop_first=True)
    
    X = df.drop(columns=['price'])
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    preprocess_data('data/house_prices.csv')




import xgboost as xgb
import pandas as pd
from sklearn.metrics import mean_squared_error
import pickle

from data_preprocessing import preprocess_data

def train_xgboost_model():
    X_train, X_test, y_train, y_test = preprocess_data('data/house_prices.csv')
    
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"XGBoost RMSE: {rmse}")
    
    # Save the model
    with open('models/house_price_xgboost_model.pkl', 'wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    train_xgboost_model()





import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error
import pandas as pd
from data_preprocessing import preprocess_data

def train_ann_model():
    X_train, X_test, y_train, y_test = preprocess_data('data/house_prices.csv')
    
    # Build the ANN model
    model = Sequential([
        Dense(64, input_dim=X_train.shape[1], activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)  # Output layer
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
    
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"ANN RMSE: {rmse}")
    
    # Save the model
    model.save('models/house_price_ann_model.h5')

if __name__ == "__main__":
    train_ann_model()




from flask import Flask, request, jsonify
import numpy as np
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load models
with open('models/house_price_xgboost_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

ann_model = load_model('models/house_price_ann_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([data['feature_values']])
    
    # Predict with XGBoost
    xgb_prediction = xgb_model.predict(features)
    
    # Predict with ANN
    ann_prediction = ann_model.predict(features)
    
    return jsonify({
        'xgboost_prediction': xgb_prediction[0],
        'ann_prediction': ann_prediction[0][0]
    })

if __name__ == "__main__":
    app.run(debug=True)

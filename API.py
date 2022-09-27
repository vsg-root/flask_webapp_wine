import pickle
from flask import Flask, request
import numpy as np
import pandas as pd
from wine_quality.Wine_Quality import WineQuality
import os

model = pickle.load(open('model/modelo1_wine.pkl', 'rb'))

# instanciate Flask
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    test_json = request.get_json()
    
    # Collect data
    if test_json:
        if isinstance(test_json, dict):
            df_raw = pd.DataFrame(test_json, index=[0])
        else:
            df_raw = pd.DataFrame(test_json, columns=test_json[0].keys())
    
    # Instanciate data preparation
    pipeline = WineQuality()
    
    # Data preparation
    df1 = pipeline.data_preparation(df_raw)
    
    # Predition
    pred = model.predict(df1)
    df1['prediction'] = pred
    return df1.to_json(orient='records')

if __name__ == "__main__":
    port = os.environ.get('PORT', 5000)
    app.run(host='0.0.0.0', port='5000')
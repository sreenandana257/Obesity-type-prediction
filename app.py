from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)
print(app.template_folder)
model = load_model("obesity_ann_model.h5")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")

target_encoder = encoders["NObeyesdad"]
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    data = request.form

    row = {
        'Age': float(data['Age']),
        'Gender': data['Gender'],
        'Height': float(data['Height']),
        'Weight': float(data['Weight']),
        'CALC': data['CALC'],
        'FAVC': data['FAVC'],
        'FCVC': float(data['FCVC']),
        'NCP': float(data['NCP']),
        'SCC': data['SCC'],
        'SMOKE': data['SMOKE'],
        'CH2O': float(data['CH2O']),
        'family_history_with_overweight': data['FamilyHistory'],
        'FAF': float(data['FAF']),
        'TUE': float(data['TUE']),
        'CAEC': data['CAEC'],
        'MTRANS': data['MTRANS']
    }

    df = pd.DataFrame([row])

    for col in df.columns:
        if col in encoders and col != "NObeyesdad":
            df[col] = encoders[col].transform(df[col])

    feature_order = [
        'Age',
        'Gender',
        'Height',
        'Weight',
        'CALC',
        'FAVC',
        'FCVC',
        'NCP',
        'SCC',
        'SMOKE',
        'CH2O',
        'family_history_with_overweight',
        'FAF',
        'TUE',
        'CAEC',
        'MTRANS'
    ]

    df = df[feature_order]

    scaled = scaler.transform(df)

    pred = model.predict(scaled)
    pred_class = np.argmax(pred)

    obesity = target_encoder.inverse_transform([pred_class])[0]

    return render_template(
        "index.html",
        prediction=obesity
    )
    df = pd.DataFrame([row])

    # Encode categorical variables
    for col in df.columns:
        if col in encoders and col != "NObeyesdad":
            df[col] = encoders[col].transform(df[col])

    scaled = scaler.transform(df)

    pred = model.predict(scaled)
    pred_class = np.argmax(pred)

    obesity = target_encoder.inverse_transform([pred_class])[0]

    return render_template(
        "index.html",
        prediction=obesity
    )

if __name__ == "__main__":
    print(app.url_map)
    app.run(debug=True)
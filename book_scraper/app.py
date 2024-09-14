from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
from models import LinearRegression, LogisticRegression
from scalers import CustomScaler

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    linear_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def load_logistic_model(file_path, multi_class):
    with open(file_path, 'rb') as f:
        weights, biases = pickle.load(f)
    model = LogisticRegression(lr=0.01, num_iterations=1000, multi_class=multi_class)
    model.weights = weights
    model.biases = biases
    return model

logistic_model_ovr = load_logistic_model('logistic_regression_ovr.pkl', 'ovr')
logistic_model_multinomial = load_logistic_model('logistic_regression_multinomial.pkl', 'multinomial')

def encode_categorical_features(df, categorical_columns):
    encoded_df = df.copy()
    for column in categorical_columns:
        encoded_df[column] = encoded_df[column].astype('category').cat.codes
    return encoded_df

categorical_columns = ['author', 'publisher', 'genre', 'year', 'binding']

def preprocess_input(data):
    df = pd.DataFrame([data])
    df_encoded = encode_categorical_features(df, categorical_columns)

    X = df_encoded[['author', 'publisher', 'genre', 'year', 'binding', 'pages', 'format']]

    X = X.apply(pd.to_numeric, errors='coerce')

    if pd.api.types.is_numeric_dtype(X['pages']):
        X['pages'] = np.log1p(X['pages'])
    
    if pd.api.types.is_numeric_dtype(X['format']):
        X['format'] = np.log1p(X['format'])

    X_standardized = scaler.transform(X)
    return X_standardized

def predict_logistic_regression(X, weights, biases, multi_class):
    if multi_class == 'ovr':
        num_classes = weights.shape[0]
        predictions = np.zeros((X.shape[0], num_classes))
        
        for i in range(num_classes):
            Z = np.dot(X, weights[i]) + biases[i]
            A = 1 / (1 + np.exp(-Z)) 
            predictions[:, i] = A
        
        return np.argmax(predictions, axis=1)
    
    elif multi_class == 'multinomial':
        Z = np.dot(X, weights.T) + biases
        A = 1 / (1 + np.exp(-Z))
        return np.argmax(A, axis=1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    model_choice = request.form.get('model_choice', None)
    
    if not model_choice:
        return render_template('result.html', prediction="Model choice is required.")
    
    data = {
        'author': request.form['author'],
        'publisher': request.form['publisher'],
        'genre': request.form['genre'],
        'year': int(request.form['year']),
        'pages': int(request.form['pages']),
        'binding': request.form['binding'],
        'format': request.form['format']
    }
    
    X = preprocess_input(data)

    if model_choice == 'linear_regression':
        predicted_price = linear_model.predict(X)[0]
        return render_template('result.html', prediction=f"Predicted Price: RSD {predicted_price}")

    elif model_choice == 'logistic_regression_ovr':
        y_pred = predict_logistic_regression(X, logistic_model_ovr.weights, logistic_model_ovr.biases, multi_class='ovr')[0]
        price_ranges = {
            0: "Knjiga pripada cjenovnom rangu RSD 0-847",
            1: "Knjiga pripada cjenovnom rangu RSD 848-1250",
            2: "Knjiga pripada cjenovnom rangu skupljem od RSD 1250"
        }
        price_range_description = price_ranges.get(y_pred, "Nepoznata cjenovna kategorija")
        return render_template('result.html', prediction=f"Predviđena cjenovna kategorija (One-vs-Rest): {price_range_description}")

    elif model_choice == 'logistic_regression_multinomial':
        y_pred = predict_logistic_regression(X, logistic_model_multinomial.weights, logistic_model_multinomial.biases, multi_class='multinomial')[0]
        price_ranges = {
            0: "Knjiga pripada cjenovnom rangu RSD 0-847",
            1: "Knjiga pripada cjenovnom rangu RSD 848-1250",
            2: "Knjiga pripada cjenovnom rangu skupljem od RSD 1250"
        }
        price_range_description = price_ranges.get(y_pred, "Nepoznata cjenovna kategorija")
        return render_template('result.html', prediction=f"Predviđena cjenovna kategorija (Multinomial): {price_range_description}")

    return render_template('result.html', prediction="Invalid Model Choice")



if __name__ == '__main__':
    app.run(debug=True)
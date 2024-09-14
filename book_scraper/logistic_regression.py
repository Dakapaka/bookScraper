import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from models import LogisticRegression
import seaborn as sns
import pickle

def get_mysql_data():
    engine = create_engine('mysql+mysqlconnector://book_user:1234@localhost:3306/books_db')
    query = 'SELECT * FROM books_preprocessed'
    df = pd.read_sql(query, engine)
    return df

df = get_mysql_data()
df.fillna(0, inplace=True)

def categorize_price(price):
    if price <= 850:
        return 0
    elif price > 850 and price <= 1250:
        return 1
    else:
        return 2

df['price_category'] = df['price'].apply(categorize_price)

for column in ['pages', 'format', 'price']:
    df[column] = pd.to_numeric(df[column], errors='coerce')

def encode_categorical_features(df, categorical_columns):
    encoded_df = df.copy()
    for column in categorical_columns:
        encoded_df[column] = encoded_df[column].astype('category').cat.codes
    return encoded_df

categorical_columns = ['author', 'publisher', 'genre', 'year', 'binding']
df_encoded = encode_categorical_features(df, categorical_columns)

X = df_encoded[['author', 'publisher', 'genre', 'year', 'binding', 'pages', 'format']]
y = df_encoded['price_category']

def preprocess_features(X):
    X = X.apply(pd.to_numeric, errors='coerce')
    
    if pd.api.types.is_numeric_dtype(X['pages']):
        X['pages'] = np.log1p(X['pages'])
    
    if pd.api.types.is_numeric_dtype(X['format']):
        X['format'] = np.log1p(X['format'])
    
    mean = X.mean()
    std = X.std()
    X_standardized = (X - mean) / std
    return X_standardized

X = preprocess_features(X)

def train_test_split(X, y, test_size=0.2):
    n = X.shape[0]
    test_size = int(n * test_size)
    indices = np.random.permutation(n)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split(X, y)

def train_model(X_train, y_train, multi_class):
    model = LogisticRegression(lr=0.01, num_iterations=1000, multi_class=multi_class)
    model.fit(X_train, y_train)
    return model

model_ovr = train_model(X_train, y_train, 'ovr')
model_multinomial = train_model(X_train, y_train, 'multinomial')

def predict_and_evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    num_classes = len(np.unique(y_test))
    confusion_matrix = np.zeros((num_classes, num_classes))
    
    for true, pred in zip(y_test, y_pred):
        confusion_matrix[true, pred] += 1
    
    print("Confusion Matrix:")
    print(confusion_matrix)
    
    for i in range(num_classes):
        TP = confusion_matrix[i, i]
        FN = np.sum(confusion_matrix[i, :]) - TP
        FP = np.sum(confusion_matrix[:, i]) - TP
        TN = np.sum(confusion_matrix) - (TP + FP + FN)
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Class {i}:")
        print(f"  True Positives (TP): {TP}")
        print(f"  False Positives (FP): {FP}")
        print(f"  True Negatives (TN): {TN}")
        print(f"  False Negatives (FN): {FN}")
        print(f"  Precision: {precision:.2f}")
        print(f"  Recall: {recall:.2f}")
        print(f"  F1 Score: {f1_score:.2f}")

print("\nOne-vs-Rest Logistic Regression:")
predict_and_evaluate(model_ovr, X_test, y_test)

print("\nMultinomial Logistic Regression:")
predict_and_evaluate(model_multinomial, X_test, y_test)

def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump((model.weights, model.biases), f)

save_model(model_ovr, 'logistic_regression_ovr.pkl')
save_model(model_multinomial, 'logistic_regression_multinomial.pkl')

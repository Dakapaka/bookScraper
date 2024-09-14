import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import pickle
from models import LinearRegression
from scalers import CustomScaler

def get_mysql_data():
    engine = create_engine('mysql+mysqlconnector://book_user:1234@localhost:3306/books_db')
    query = 'SELECT * FROM books_preprocessed'
    df = pd.read_sql(query, engine)
    return df

df = get_mysql_data()
df.fillna(0, inplace=True)

# Remove rows where price is more than 3000
df = df[df['price'] < 3000]

# Ensure columns are of correct type
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
X = X.apply(pd.to_numeric, errors='coerce')
y = df_encoded['price']

def preprocess_features(X):
    X = X.apply(pd.to_numeric, errors='coerce')
    
    if pd.api.types.is_numeric_dtype(X['pages']):
        X['pages'] = np.log1p(X['pages'])
    
    if pd.api.types.is_numeric_dtype(X['format']):
        X['format'] = np.log1p(X['format'])

    if pd.api.types.is_numeric_dtype(X['binding']):
        X['binding'] = np.log1p(X['binding'])
    
    scaler = CustomScaler()
    X_standardized = scaler.fit_transform(X)
    return X_standardized, scaler

X, scaler = preprocess_features(X)

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

model = LinearRegression(lr=0.00045, n_iters=30000, lambda_=0.01)
model.fit(X_train, y_train)

y_predicted = model.predict(X_test)
score = model.score(X_test, y_test)
rmse = np.sqrt(np.sum((y_test - y_predicted) ** 2) / len(y_test))

print(f'Model score: {score}')
print(f'Root Mean Square Error: {rmse}')

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

if len(y_test) > 0 and len(y_predicted) > 0:
    plt.figure(figsize=(8, 6))

    plt.scatter(y_test, y_predicted, color='blue', alpha=0.7, label='Predicted vs True')

    min_value = min(y_test.min(), y_predicted.min())
    max_value = max(y_test.max(), y_predicted.max())
    plt.plot([min_value, max_value], [min_value, max_value], '--', color='red', label='Ideal Line')

    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True Values vs. Predicted Values')
    plt.savefig('plots/models_results/linear_regression.png')
    plt.legend()

else:
    print("No data to plot")

import seaborn as sns

plt.figure(figsize=(10, 10))
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.savefig('plots/models_results/correlation_matrix.png')
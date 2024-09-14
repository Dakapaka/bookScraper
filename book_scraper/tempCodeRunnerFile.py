import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from models import LogisticRegression
import seaborn as sns
import pickle

# Load data from MySQL
def get_mysql_data():
    engine = create_engine('mysql+mysqlconnector://book_user:1234@localhost:3306/books_db')
    query = 'SELECT * FROM books_preprocessed'
    df = pd.read_sql(query, engine)
    return df

# Load the data
df = get_mysql_data()
df.fillna(0, inplace=True)

# Categorize the prices
def categorize_price(price):
    if price <= 847:
        return 0
    elif price > 847 and price <= 1250:
        return 1
    else:
        return 2

df['price_category'] = df['price'].apply(categorize_price)

# Ensure columns are of correct type
for column in ['pages', 'format', 'price']:
    df[column] = pd.to_numeric(df[column], errors='coerce')

# Define target encoding function
def target_encode(df, target_col, categorical_columns):
    df_encoded = df.copy()
    for col in categorical_columns:
        # Calculate mean of the target for each category
        means = df_encoded.groupby(col)[target_col].mean()
        df_encoded[col] = df_encoded[col].map(means)
    return df_encoded

# Apply target encoding for price_category (classification task)
categorical_columns = ['author', 'publisher', 'genre', 'year', 'binding']
df_encoded = target_encode(df, 'price_category', categorical_columns)

# Split the data into features and target
X = df_encoded[['author', 'publisher', 'genre', 'year', 'binding', 'pages', 'format']]
y = df_encoded['price_category']

# Preprocess features
def preprocess_features(X):
    X = X.apply(pd.to_numeric, errors='coerce')
    
    # Apply log transformation and handle zeros
    if pd.api.types.is_numeric_dtype(X['pages']):
        X['pages'] = np.log1p(X['pages'])
    
    if pd.api.types.is_numeric_dtype(X['format']):
        X['format'] = np.log1p(X['format'])
    
    # Standardize features
    mean = X.mean()
    std = X.std()
    X_standardized = (X - mean) / std
    return X_standardized

X = preprocess_features(X)

# Split the data into training and testing sets
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

# Train models with adjusted hyperparameters
def train_model(X_train, y_train, multi_class):
    model = LogisticRegression(lr=0.01, num_iterations=1000, multi_class=multi_class)
    model.fit(X_train, y_train)
    return model

model_ovr = train_model(X_train, y_train, 'ovr')
model_multinomial = train_model(X_train, y_train, 'multinomial')

# Predict and evaluate models
def predict_and_evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    num_classes = len(np.unique(y_test))
    confusion_matrix = np.zeros((num_classes, num_classes))
    
    # Compute confusion matrix
    for true, pred in zip(y_test, y_pred):
        confusion_matrix[true, pred] += 1
    
    # Print confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix)
    
    # Print detailed metrics for each class
    for i in range(num_classes):
        TP = confusion_matrix[i, i]
        FN = np.sum(confusion_matrix[i, :]) - TP
        FP = np.sum(confusion_matrix[:, i]) - TP
        TN = np.sum(confusion_matrix) - (TP + FP + FN)
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        accuracy = (TP + TN) / np.sum(confusion_matrix)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nClass {i}:")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"F1 Score: {f1:.2f}")

    return confusion_matrix

confusion_matrix_ovr = predict_and_evaluate(model_ovr, X_test, y_test)
confusion_matrix_multinomial = predict_and_evaluate(model_multinomial, X_test, y_test)

# Save the models
with open('logistic_regression_ovr.pkl', 'wb') as f:
    pickle.dump(model_ovr, f)

with open('logistic_regression_multinomial.pkl', 'wb') as f:
    pickle.dump(model_multinomial, f)

# Plot the correlation matrix
plt.figure(figsize=(10, 10))
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

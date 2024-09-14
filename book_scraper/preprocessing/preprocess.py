import mysql.connector
import pandas as pd
from sqlalchemy import create_engine

def get_mysql_data():
    conn = mysql.connector.connect(
        host='localhost',
        database='books_db',
        user='book_user',
        password='1234',
        port=3306
    )

    query = 'SELECT * FROM books_original'
    df = pd.read_sql(query, conn)
    conn.close()

    return df

df = get_mysql_data()

def calculate_area(row):
        if 'x' in row:
            parts = row.split('x')
            if len(parts) == 2:
                try:
                    return float(parts[0]) * float(parts[1])
                except ValueError:
                    return 0
        return 0

def preprocess_data(df):
    
    df = df.dropna(subset=['title', 'author', 'genre', 'publisher', 'year', 'pages', 'binding', 'format', 'description'])
    
    df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0.0)
    df = df[df['price'] > 0]
    df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
    df = df[(df['year'] >= 1900) & (df['year'] <= 2024)]
    df['pages'] = pd.to_numeric(df['pages'], errors='coerce').fillna(0).astype(int)
    df['binding'] = df['binding'].fillna('Unknown')
    df['genre'] = df['genre'].str.split('/').str[0].str.capitalize()
    df['format'] = df['format'].fillna('Unknown').str.replace(' cm', '').str.replace(',', '.').apply(calculate_area)
    df = df[df['format'] > 0]
    df['format'] = df['format'].astype(str)

    for col in ['title', 'publisher']:
        df[col] = df[col].str.capitalize()

    df = df.drop_duplicates(subset=['title', 'author', 'publisher'])

    df['description'] = df['description'].str.lower().str.replace('[^\w\s]', '').str.split().apply(lambda x: ' '.join(sorted(set(x), key=x.index)))
    df = df.drop_duplicates(subset=['description'])

    #only Broš, Tvrd and Mek should be present in the dataset
    df = df[(df['binding'] == 'Broš') | (df['binding'] == 'Tvrd') | (df['binding'] == 'Mek')]

    return df


df = preprocess_data(df)

def save_to_mysql(df):
    conn = mysql.connector.connect(
        host='localhost',
        database='books_db',
        user='book_user',
        password='1234',
        port=3306
    )

    engine = create_engine('mysql+mysqlconnector://book_user:1234@localhost:3306/books_db')
    df.to_sql('books_preprocessed', con=engine, if_exists='replace', index=False, chunksize=1000)

    conn.close()

save_to_mysql(df)

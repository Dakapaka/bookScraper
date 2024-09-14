import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_mysql_connection():
    return mysql.connector.connect(
        host='localhost',
        database='books_db',
        user='book_user',
        password='1234',
        port=3306
    )

def execute_query(query):
    conn = get_mysql_connection()
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def top_10_publishers():
    query = """
    SELECT publisher, COUNT(*) AS broj
    FROM books_preprocessed
    GROUP BY publisher
    ORDER BY broj DESC
    LIMIT 10;
    """
    return execute_query(query)

def books_by_genre():
    query = """
    SELECT genre, COUNT(*) AS broj
    FROM books_preprocessed
    GROUP BY genre
    ORDER BY broj DESC;
    """
    return execute_query(query)

def books_by_decades():
    query = """
    SELECT 
        CASE 
            WHEN year BETWEEN 1961 AND 1970 THEN '1961-1970'
            WHEN year BETWEEN 1971 AND 1980 THEN '1971-1980'
            WHEN year BETWEEN 1981 AND 1990 THEN '1981-1990'
            WHEN year BETWEEN 1991 AND 2000 THEN '1991-2000'
            WHEN year BETWEEN 2001 AND 2010 THEN '2001-2010'
            WHEN year BETWEEN 2011 AND 2020 THEN '2011-2020'
            ELSE '2021-present'
        END AS decade,
        COUNT(*) AS broj
    FROM books_preprocessed
    GROUP BY decade
    ORDER BY decade;
    """
    return execute_query(query)

def books_by_top_5_publishers():
    query = """
    SELECT publisher, COUNT(*) AS broj,
           ROUND((COUNT(*) / (SELECT COUNT(*) FROM books_preprocessed) * 100), 2) AS percentage
    FROM books_preprocessed
    GROUP BY publisher
    ORDER BY broj DESC
    LIMIT 5;
    """
    return execute_query(query)

def books_by_price_range():
    query = """
    SELECT 
        CASE 
            WHEN price <= 500 THEN '≤ 500 dinars'
            WHEN price BETWEEN 501 AND 1500 THEN '501-1500 dinars'
            WHEN price BETWEEN 1501 AND 3000 THEN '1501-3000 dinars'
            WHEN price BETWEEN 3001 AND 5000 THEN '3001-5000 dinars'
            WHEN price BETWEEN 5001 AND 10000 THEN '5001-10000 dinars'
            WHEN price BETWEEN 10001 AND 15000 THEN '10001-15000 dinars'
            ELSE '15001 or more dinars'
        END AS price_range,
        COUNT(*) AS broj,
        ROUND((COUNT(*) / (SELECT COUNT(*) FROM books_preprocessed) * 100), 2) AS percentage
    FROM books_preprocessed
    GROUP BY price_range
    ORDER BY price_range;
    """
    return execute_query(query)

def books_with_hard_covers_vs_all():
    query = """
    SELECT 
        (SELECT COUNT(*) FROM books_preprocessed WHERE binding = 'Tvrd' AND year BETWEEN 2021 AND 2024) AS hard_cover_books,
        (SELECT COUNT(*) FROM books_preprocessed WHERE year BETWEEN 2021 AND 2024) AS total_books;
    """
    return execute_query(query)

def visualize_top_10_publishers():
    df = top_10_publishers()
    plt.figure(figsize=(10, 6))
    sns.barplot(x='broj', y='publisher', data=df, palette='viridis')
    plt.title('Top 10 Izdavaca sa najvise knjiga')
    plt.xlabel('Broj knjiga')
    plt.ylabel('Izdavac')
    #plt.show()
    plt.savefig('plots/statistics/top_10_publishers.png')

def visualize_books_by_genre():
    df = books_by_genre()
    plt.figure(figsize=(12, 6))
    sns.barplot(x='broj', y='genre', data=df, palette='inferno')
    plt.title('Broj knjiga po zanru')
    plt.xlabel('Broj knjiga')
    plt.ylabel('Zanr')
    #plt.show()
    plt.savefig('plots/statistics/books_by_genre.png')

def visualize_books_by_decades():
    df = books_by_decades()
    plt.figure(figsize=(10, 6))
    sns.barplot(x='decade', y='broj', data=df, palette='magma')
    plt.title('Broj knjiga izdatih po decenijama')
    plt.xlabel('Decenija')
    plt.ylabel('Broj knjiga')
    #plt.show()
    plt.savefig('plots/statistics/books_by_decades.png')

def visualize_books_by_top_5_publishers():
    df = books_by_top_5_publishers()
    plt.figure(figsize=(10, 6))
    sns.barplot(x='percentage', y='publisher', data=df, palette='coolwarm')
    plt.title('Top 5 Izdavaca sa najvise knjiga(%)') 
    plt.xlabel('Procenat knjiga')
    plt.ylabel('Izdavac')
    #plt.show()
    plt.savefig('plots/statistics/books_by_top_5_publishers.png')

def visualize_books_by_price_range():
    df = books_by_price_range()
    plt.figure(figsize=(12, 6))
    sns.barplot(x='percentage', y='price_range', data=df, palette='Spectral')
    plt.title('Knjige po cjenovnom rangu(%)')
    plt.xlabel('Procenat knjiga')
    plt.ylabel('Cjenovni rang')
    #plt.show()
    plt.savefig('plots/statistics/books_by_price_range.png')

def visualize_books_with_hard_covers_vs_all():
    df = books_with_hard_covers_vs_all()
    hard_cover_books = df['hard_cover_books'][0]
    total_books = df['total_books'][0]
    other_books = total_books - hard_cover_books

    hard_cover_percentage = round((hard_cover_books / total_books) * 100, 2)
    
    labels = [f'Knjige sa tvrdim povezom ({hard_cover_percentage}%)', 'Ostale knjige']
    sizes = [hard_cover_books, other_books]
    colors = ['#ff9999', '#66b3ff']
    
    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.title('Procenat knjiga sa tvrdim povezom u poslednje 3 godine')
    plt.axis('equal')
    #plt.show()
    plt.savefig('plots/statistics/books_with_hard_covers_vs_all.png')

visualize_top_10_publishers()
visualize_books_by_genre()
visualize_books_by_decades()
visualize_books_by_top_5_publishers()
visualize_books_by_price_range()
visualize_books_with_hard_covers_vs_all()

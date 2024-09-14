import mysql.connector
from scrapy.exceptions import NotConfigured

class MySQLPipeline:
    def open_spider(self, spider):
        host = spider.settings.get('MYSQL_HOST')
        database = spider.settings.get('MYSQL_DATABASE')
        user = spider.settings.get('MYSQL_USER')
        password = spider.settings.get('MYSQL_PASSWORD')
        port = spider.settings.get('MYSQL_PORT')

        if not all([host, database, user, password, port]):
            raise NotConfigured("MySQL settings not configured.")

        self.conn = mysql.connector.connect(
            host=host,
            database=database,
            user=user,
            password=password,
            port=port
        )
        self.cursor = self.conn.cursor()

    def close_spider(self, spider):
        self.conn.close()

    def process_item(self, item, spider):

        price = item.get('price')
        if price is None:
            price = 0.0 

        try:
            price = float(price)
        except ValueError:
            price = 0.0

        self.cursor.execute("""
            INSERT INTO books (title, author, genre, publisher, year, pages, binding, format, description, price)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            item['title'],
            item['author'],
            item['genre'],
            item['publisher'],
            item['year'],
            item['pages'],
            item['binding'],
            item['format'],
            item['description'],
            price
        ))
        self.conn.commit()
        return item

import scrapy
import re

class VulkanSpider(scrapy.Spider):
    name = 'vulkan'
    allowed_domains = ['knjizare-vulkan.rs']
    start_urls = ['https://www.knjizare-vulkan.rs/domaca-decja-literatura']

    def start_requests(self):
        for page in range(1, 244):
            url = f'https://www.knjizare-vulkan.rs/domaca-decja-literatura/page-{page}'
            yield scrapy.Request(url, callback=self.parse)

    def parse(self, response):
        if response.status == 400:
            self.logger.error(f"Bad Request: {response.url}")
            return

        book_links = response.css('a.product-link::attr(href)').getall()
        for link in book_links:
            yield response.follow(link, self.parse_book)

    def parse_book(self, response):
        if response.status == 400:
            self.logger.error(f"Bad Request on Book Page: {response.url}")
            self.logger.error(f"Response body: {response.text}")
            return

        title = response.css('div.title h1 span::text').get().strip()
        self.logger.info(f"Title: {title}")

        author = response.css('div.item span.value a::text').get().strip()
        self.logger.info(f"Author: {author}")

        genre = response.css('div.category a::text').get().strip()
        self.logger.info(f"Genre: {genre}")

        publisher = response.css('tr.attr-izdavac td:nth-child(2) a::text').get().strip()
        self.logger.info(f"Publisher: {publisher}")

        year = response.css('tr td:contains("Godina") + td::text').get().strip()
        self.logger.info(f"Year: {year}")

        pages = response.css('tr td:contains("Strana") + td::text').get().strip()
        self.logger.info(f"Pages: {pages}")

        binding = response.css('tr.attr-povez td:nth-child(2)::text').get().strip()
        self.logger.info(f"Binding: {binding}")

        format = response.css('tr td:contains("Format") + td::text').get().strip()
        self.logger.info(f"Format: {format}")

        description = response.css('div.tab-pane#tab_product_description::text').getall()
        description = ' '.join(part.strip() for part in description).replace('<br>', ' ').strip()
        self.logger.info(f"Description: {description}")

        price_text = response.css('span.product-price-without-discount-value::text').get().strip()
        price = re.sub(r'[^\d,]', '', price_text).replace(',', '.')
        self.logger.info(f"Price: {price}")

        yield {
            'title': title,
            'author': author,
            'genre': genre,
            'publisher': publisher,
            'year': year,
            'pages': pages,
            'binding': binding,
            'format': format,
            'description': description,
            'price': price,
        }

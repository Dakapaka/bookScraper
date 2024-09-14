import scrapy
from scrapy import Request
import time
import random
import re

class LagunaSpider(scrapy.Spider):
    name = 'laguna'
    allowed_domains = ['laguna.rs']
    start_urls = [
        'https://www.laguna.rs/z9_zanr_akcioni_laguna.html',
        'https://www.laguna.rs/z93_zanr_autobiografija_laguna.html',
        'https://www.laguna.rs/z25_zanr_avanturisticki_laguna.html',
        'https://www.laguna.rs/z92_zanr_biografija_laguna.html',
        'https://www.laguna.rs/z140_zanr_bojanke_laguna.html',
        'https://www.laguna.rs/z146_zanr_bojanke_za_odrasle_laguna.html',
        'https://www.laguna.rs/z134_zanr_ciklit_laguna.html',
        'https://www.laguna.rs/z15_zanr_domaci_autori_laguna.html',
        'https://www.laguna.rs/z6_zanr_drama_laguna.html',
        'https://www.laguna.rs/z154_zanr_drustvene_igre_laguna.html',
        'https://www.laguna.rs/z139_zanr_duh_i_telo_laguna.html',
        'https://www.laguna.rs/z136_zanr_e-knjige_laguna.html',
        'https://www.laguna.rs/z31_zanr_edukativni_laguna.html',
        'https://www.laguna.rs/z75_zanr_ekranizovane_knjige_laguna.html',
        'https://www.laguna.rs/z98_zanr_enciklopedija_laguna.html',
        'https://www.laguna.rs/z24_zanr_epska_fantastika_laguna.html',
        'https://www.laguna.rs/z138_zanr_erotski_laguna.html',
        'https://www.laguna.rs/z55_zanr_fantastika_laguna.html',
        'https://www.laguna.rs/z128_zanr_filozofija_laguna.html',
        'https://www.laguna.rs/z12_zanr_horor_laguna.html',
        'https://www.laguna.rs/z114_zanr_interaktivna_knjiga_laguna.html',
        'https://www.laguna.rs/z149_zanr_internet_i_racunari_laguna.html',
        'https://www.laguna.rs/z150_zanr_istorija_laguna.html',
        'https://www.laguna.rs/z5_zanr_istorijski_laguna.html',
        'https://www.laguna.rs/z22_zanr_istorijski_triler_laguna.html',
        'https://www.laguna.rs/z148_zanr_klasici_laguna.html',
        'https://www.laguna.rs/z32_zanr_knjige_za_decu_laguna.html',
        'https://www.laguna.rs/z10_zanr_komedija_laguna.html',
        'https://www.laguna.rs/z26_zanr_komicna_fantastika_laguna.html',
        'https://www.laguna.rs/z27_zanr_kriminalisticki_laguna.html',
        'https://www.laguna.rs/z30_zanr_kuvari_laguna.html',
        'https://www.laguna.rs/z7_zanr_ljubavni_laguna.html',
        'https://www.laguna.rs/z153_zanr_manga_i_strip_laguna.html',
        'https://www.laguna.rs/z137_zanr_marketing_i_menadzment_laguna.html',
        'https://www.laguna.rs/z107_zanr_mitologije_laguna.html',
        'https://www.laguna.rs/z142_zanr_muzika_laguna.html',
        'https://www.laguna.rs/z4_zanr_nagradjene_knjige_laguna.html',
        'https://www.laguna.rs/z11_zanr_naucna_fantastika_laguna.html',
        'https://www.laguna.rs/z143_zanr_poezija_laguna.html',
        'https://www.laguna.rs/z19_zanr_popularna_nauka_laguna.html',
        'https://www.laguna.rs/z39_zanr_popularna_psihologija_laguna.html',
        'https://www.laguna.rs/z151_zanr_potpisane_knjige_laguna.html',
        'https://www.laguna.rs/z124_zanr_price_laguna.html',
        'https://www.laguna.rs/z18_zanr_psihologija_laguna.html',
        'https://www.laguna.rs/z89_zanr_publicistika_laguna.html',
        'https://www.laguna.rs/z21_zanr_putopisi_laguna.html',
        'https://www.laguna.rs/z145_zanr_sa_nalepnicama_laguna.html',
        'https://www.laguna.rs/z33_zanr_slikovnice_laguna.html',
        'https://www.laguna.rs/z152_zanr_specijalna_ponuda_laguna.html',
        'https://www.laguna.rs/z147_zanr_sport_laguna.html',
        'https://www.laguna.rs/z141_zanr_teorije_zavere_laguna.html',
        'https://www.laguna.rs/z13_zanr_tinejdz_laguna.html',
        'https://www.laguna.rs/z8_zanr_trileri_laguna.html',
        'https://www.laguna.rs/z29_zanr_umetnost_laguna.html',
        'https://www.laguna.rs/z14_zanr_za_poklon_laguna.html'
    ]

    def parse(self, response):
        if response.status == 400:
            self.logger.error(f"Bad Request: {response.url}")
            return
        
        book_links = response.css('div.knjiga a.naslov::attr(href)').getall()
        for link in book_links:
            yield response.follow(link, self.parse_book)

        next_page = response.css('a.sledeca::attr(href)').get()
        if next_page:
            yield response.follow(next_page, self.parse)

    def parse_book(self, response):
        if response.status == 400:
            self.logger.error(f"Bad Request on Book Page: {response.url}")
            self.logger.error(f"Response body: {response.text}")
            return

        title = response.css('h1.naslov::text').get()
        self.logger.info(f"Title: {title}")
        
        author = response.css('h2 a::text').get()
        self.logger.info(f"Author: {author}")
        
        genre_elements = response.css('div.podatak h3 a::text').getall()
        genre = ' / '.join([g.strip() for g in genre_elements]) if genre_elements else 'Not Available'
        self.logger.info(f"Genre: {genre}")
        
        publisher = "Laguna"
        self.logger.info(f"Publisher: {publisher}")
        
        year_text = response.xpath('//span[contains(text(), "Godina izdanja")]/following-sibling::text()[1]').get(default='Not Available').strip()
        year_match = re.search(r'\b(\d{4})\b', year_text)
        year = year_match.group(1) if year_match else '0000'
        self.logger.info(f"Year: {year}")
        
        pages_text = response.xpath('//span[contains(text(), "Broj strana:")]/following-sibling::text()').get(default='Not Available').strip()
        pages = re.sub(r'[^\d]', '', pages_text) if pages_text else 'Not Available'
        self.logger.info(f"Pages: {pages}")
        
        binding = response.xpath('//span[contains(text(), "Povez:")]/following-sibling::text()').get(default='Not Available').strip()
        self.logger.info(f"Binding: {binding}")
        
        format = response.xpath('//span[contains(text(), "Format:")]/following-sibling::text()').get(default='Not Available').strip()
        self.logger.info(f"Format: {format}")
        
        description = ' '.join(response.css('div#oknjizitab div *::text').getall()).strip()
        self.logger.info(f"Description: {description}")
        
        price_text = response.css('div.cena p::text').get()
        self.logger.info(f"Price text: {price_text}")
        
        match = re.search(r'(\d+[\.,]\d+)', price_text or '')
        if match:
            price = match.group(1).replace(',', '.')
        else:
            price = None
        
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

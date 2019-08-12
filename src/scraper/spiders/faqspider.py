import scrapy 
import re
import html2text
import json
  
class FaqspiderSpider(scrapy.Spider): 
    name = "faqspider"
  
    # request function 
    def start_requests(self): 
        # urls = [ 'https://www.credit-suisse.com/lu/en/private-banking/services/online-banking/faq/sicherheit-login.html', ] 
        # urls = [ 'https://www.credit-suisse.com/lu/en/private-banking/services/online-banking/faq.html', ] 
        urls = [ 'https://www.credit-suisse.com/lu/en/private-banking/services/online-banking/faq/einstellungen.html', ] 
          
        for url in urls: 
            yield scrapy.Request(url = url, callback = self.parse) 
  
    # Parse function 
    def parse(self, response): 
          
        question = response.xpath('//a[@class="mod_content_accordion_title_link"]').extract()

        p = response.xpath('//div[@class="mod_content_accordion_tab_panel"]//div[@class="component_standard"]//div[@class="component_standard_content"]//div[@class="mod_text_component"]').extract()

        count = 0
        arr = []
        for idx, x in enumerate(question):
            count = count + 1
            temp_string = ""
            for t in html2text.html2text(p[idx]).split('\n'):
                temp_string  = temp_string + t
       
            arr.append({
                "MESSAGE": "" + html2text.html2text(x).rstrip('\n') + "",
                "RESPONSE": "" + temp_string + ""
            })
        print(arr)
        with open('lsa/links.json', "w") as jsonFile:
            jsonFile.write(json.dumps(arr, indent = 4))

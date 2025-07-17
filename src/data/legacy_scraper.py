
import scrapy
import pandas as pd
from scrapy.utils.response import open_in_browser
from scrapy import FormRequest

class PricesSpider(scrapy.Spider):
    name = "prices"
    
    start_urls = ["https://josaa.admissions.nic.in/applicant/seatmatrix/openingclosingrankarchieve.aspx"]

    def parse(self, response):
        # Loop through the years from 2023 to 2016
        for year in range(2023, 2015, -1):
            data = {
                "ctl00$hdnSecKey": "",
                "ctl00$ContentPlaceHolder1$ddlYear": str(year)
            }
            yield FormRequest.from_response(response, formdata=data, meta={'year': year, 'round': 1}, callback=self.step2)

    def step2(self, response):
        year = response.meta['year']
        round_no = response.meta['round']

        data = {
            "ctl00$ContentPlaceHolder1$ddlroundno": str(round_no)
        }

        yield FormRequest.from_response(response, formdata=data, meta={'year': year, 'round': round_no}, callback=self.step3)

    def step3(self, response):
        year = response.meta['year']
        round_no = response.meta['round']

        data = {
            "ctl00$ContentPlaceHolder1$ddlInstype": "ALL"
        }

        yield FormRequest.from_response(response, formdata=data, meta={'year': year, 'round': round_no}, callback=self.step4)

    def step4(self, response):
        year = response.meta['year']
        round_no = response.meta['round']

        data = {
            "ctl00$ContentPlaceHolder1$ddlInstitute": "ALL"
        }

        yield FormRequest.from_response(response, formdata=data, meta={'year': year, 'round': round_no}, callback=self.step5)

    def step5(self, response):
        year = response.meta['year']
        round_no = response.meta['round']

        data = {
            "ctl00$ContentPlaceHolder1$ddlBranch": "ALL"
        }

        yield FormRequest.from_response(response, formdata=data, meta={'year': year, 'round': round_no}, callback=self.step6)

    def step6(self, response):
        year = response.meta['year']
        round_no = response.meta['round']

        data = {
            "ctl00$ContentPlaceHolder1$ddlSeatType": "ALL",
            "ctl00$ContentPlaceHolder1$btnSubmit": "Submit"
        }

        yield FormRequest.from_response(response, formdata=data, meta={'year': year, 'round': round_no}, callback=self.step7)

    def step7(self, response):
        year = response.meta['year']
        round_no = response.meta['round']
        
        open_in_browser(response)  # Optional: for debugging purposes

        dfs = pd.read_html(response.text)
        for i, df in enumerate(dfs):
            df.to_csv(f"{year}_round{round_no}_{i}.csv")

        # If current round is less than 6, move to the next round for the same year
        if round_no < 6:
            next_round_no = round_no + 1
            data = {
                "ctl00$ContentPlaceHolder1$ddlroundno": str(next_round_no)
            }
            yield FormRequest.from_response(response, formdata=data, meta={'year': year, 'round': next_round_no}, callback=self.step3)
        # If current round is 6 and year is more than 2016, move to the next year and reset round to 1
        elif year > 2016:
            next_year = year - 1
            data = {
                "ctl00$ContentPlaceHolder1$ddlYear": str(next_year)
            }
            yield FormRequest.from_response(response, formdata=data, meta={'year': next_year, 'round': 1}, callback=self.step2)

    """
    def parse(self,response):
        data={
            "ctl00$ContentPlaceHolder1$ddlroundno":"",
            "ctl00$hdnSecKey": "",
            "ctl00$ContentPlaceHolder1$ddlroundno": "5"
        }
        yield FormRequest.from_response(response,formdata=data,callback=self.step2)

    def step2(self,response):
        data={
            "ctl00$ContentPlaceHolder1$ddlInstype": "ALL"
        }
        yield FormRequest.from_response(response,formdata=data,callback=self.step3)



    
    def step3(self,response):
        data={
            "ctl00$ContentPlaceHolder1$ddlInstitute": "ALL"
        }
        yield FormRequest.from_response(response,formdata=data,callback=self.step4)

    

    def step4(self,response):
        data={
            "ctl00$ContentPlaceHolder1$ddlBranch": "ALL"
        }
        yield FormRequest.from_response(response,formdata=data,callback=self.step5)


    
    def step5(self,response):
        data={
            "ctl00$ContentPlaceHolder1$ddlSeattype": "ALL"
        }
        yield FormRequest.from_response(response,formdata=data,callback=self.step6)

    def step6(self,response):
        open_in_browser(response)

        dfs=pd.read_html(response.text)
        for i,df in enumerate(dfs):
            df.to_csv(f"2024_round5_{i}.csv")
    """
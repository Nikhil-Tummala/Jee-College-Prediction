"""
JEE College Prediction Data Scraper

This module contains the web scraper for collecting JEE admission data
from the official JOSAA website.
"""

import scrapy
import pandas as pd
from scrapy.utils.response import open_in_browser
from scrapy import FormRequest
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JEEDataScraper(scrapy.Spider):
    """
    A Scrapy spider for scraping JEE admission data from JOSAA website.
    
    This spider navigates through multiple years, rounds, and filters to collect
    comprehensive admission data including opening and closing ranks for different
    institutes, branches, and seat types.
    """
    
    name = "jee_data_scraper"
    
    start_urls = ["https://josaa.admissions.nic.in/applicant/seatmatrix/openingclosingrankarchieve.aspx"]
    
    def __init__(self, start_year=2023, end_year=2016, max_rounds=6, *args, **kwargs):
        """
        Initialize the scraper with configurable parameters.
        
        Args:
            start_year (int): Starting year for data collection
            end_year (int): Ending year for data collection
            max_rounds (int): Maximum number of rounds to scrape per year
        """
        super(JEEDataScraper, self).__init__(*args, **kwargs)
        self.start_year = start_year
        self.end_year = end_year
        self.max_rounds = max_rounds
        self.output_dir = "../../data/raw"
        os.makedirs(self.output_dir, exist_ok=True)

    def parse(self, response):
        """
        Initial parsing method that starts the scraping process.
        Loops through years from start_year to end_year.
        """
        logger.info(f"Starting data collection from {self.start_year} to {self.end_year}")
        
        for year in range(self.start_year, self.end_year - 1, -1):
            data = {
                "ctl00$hdnSecKey": "",
                "ctl00$ContentPlaceHolder1$ddlYear": str(year)
            }
            yield FormRequest.from_response(
                response, 
                formdata=data, 
                meta={'year': year, 'round': 1}, 
                callback=self.select_round
            )

    def select_round(self, response):
        """
        Select the round number for the current year.
        """
        year = response.meta['year']
        round_no = response.meta['round']
        
        logger.info(f"Processing Year: {year}, Round: {round_no}")

        data = {
            "ctl00$ContentPlaceHolder1$ddlroundno": str(round_no)
        }

        yield FormRequest.from_response(
            response, 
            formdata=data, 
            meta={'year': year, 'round': round_no}, 
            callback=self.select_institute_type
        )

    def select_institute_type(self, response):
        """
        Select institute type (ALL) for the current year and round.
        """
        year = response.meta['year']
        round_no = response.meta['round']

        data = {
            "ctl00$ContentPlaceHolder1$ddlInstype": "ALL"
        }

        yield FormRequest.from_response(
            response, 
            formdata=data, 
            meta={'year': year, 'round': round_no}, 
            callback=self.select_institute
        )

    def select_institute(self, response):
        """
        Select institute (ALL) for the current selection.
        """
        year = response.meta['year']
        round_no = response.meta['round']

        data = {
            "ctl00$ContentPlaceHolder1$ddlInstitute": "ALL"
        }

        yield FormRequest.from_response(
            response, 
            formdata=data, 
            meta={'year': year, 'round': round_no}, 
            callback=self.select_branch
        )

    def select_branch(self, response):
        """
        Select branch (ALL) for the current selection.
        """
        year = response.meta['year']
        round_no = response.meta['round']

        data = {
            "ctl00$ContentPlaceHolder1$ddlBranch": "ALL"
        }

        yield FormRequest.from_response(
            response, 
            formdata=data, 
            meta={'year': year, 'round': round_no}, 
            callback=self.submit_form
        )

    def submit_form(self, response):
        """
        Submit the form with all selected parameters and scrape the data.
        """
        year = response.meta['year']
        round_no = response.meta['round']

        data = {
            "ctl00$ContentPlaceHolder1$ddlSeatType": "ALL",
            "ctl00$ContentPlaceHolder1$btnSubmit": "Submit"
        }

        yield FormRequest.from_response(
            response, 
            formdata=data, 
            meta={'year': year, 'round': round_no}, 
            callback=self.extract_data
        )

    def extract_data(self, response):
        """
        Extract and save the data from the response.
        Also handles navigation to next round or year.
        """
        year = response.meta['year']
        round_no = response.meta['round']
        
        try:
            # Extract tables from the HTML response
            dfs = pd.read_html(response.text)
            
            # Save each dataframe as a separate CSV file
            for i, df in enumerate(dfs):
                filename = f"{self.output_dir}/{year}_round{round_no}_table{i}.csv"
                df.to_csv(filename, index=False)
                logger.info(f"Saved data to {filename}")
                
        except Exception as e:
            logger.error(f"Error extracting data for {year} round {round_no}: {str(e)}")

        # Navigate to next round or year
        if round_no < self.max_rounds:
            # Move to next round for the same year
            next_round_no = round_no + 1
            data = {
                "ctl00$ContentPlaceHolder1$ddlroundno": str(next_round_no)
            }
            yield FormRequest.from_response(
                response, 
                formdata=data, 
                meta={'year': year, 'round': next_round_no}, 
                callback=self.select_institute_type
            )
        elif year > self.end_year:
            # Move to next year and reset round to 1
            next_year = year - 1
            data = {
                "ctl00$ContentPlaceHolder1$ddlYear": str(next_year)
            }
            yield FormRequest.from_response(
                response, 
                formdata=data, 
                meta={'year': next_year, 'round': 1}, 
                callback=self.select_round
            )

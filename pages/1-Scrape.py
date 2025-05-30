import streamlit as st
import logging
import json
import os
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np
import time
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.core.os_manager import ChromeType
from bs4 import BeautifulSoup
import tempfile

# Configuration dataclass
@dataclass
class ScraperConfig:
    """Configuration settings for the scraper"""
    item: str = 'Cooling shirt'
    chrome_driver_path: str = "chromedriver" 
    page_load_wait: int = 5
    min_random_wait: int = 2
    max_random_wait: int = 5
    max_pages: int = 3
    headless: bool = True
    
class Scraper:
    def __init__(self, item: str, config: ScraperConfig):
        self.config = config
        self.config.item = item
        self.item = item.replace(' ', '-')
        self.url = None
        self.soup = None
        self.product_name = []
        self.product_links = []
        self.prices = []
        self.locations = []
        self.units_sold = []
    
    def set_url(self, page_num):
        self.url = f'https://www.lazada.sg/tag/{self.item}/?page={page_num}'
    
    def scrape_site(self, page_num=1):
   
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-logging")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--allow-running-insecure-content")
        with st.echo():


            @st.cache_resource
            def get_driver():
                return webdriver.Chrome(
                    service=Service(
                        ChromeDriverManager(chrome_type=ChromeType.CHROMIUM).install()
                    ),
                    options=options,
                )


            try:
                try:
                    # service = Service()  # This will use chromedriver from PATH
                    driver = get_driver(chrome_options)
                except:
                    # Fallback to specified path if available
                    if os.path.exists(self.config.chrome_driver_path):
                        service = Service(self.config.chrome_driver_path)
                        driver = webdriver.Chrome(service=service, options=chrome_options)
                    else:
                        raise Exception("ChromeDriver not found :(")
                
                self.set_url(page_num)
                driver.get(self.url)
                time.sleep(self.config.page_load_wait)
                
                page_source = driver.page_source
                self.soup = BeautifulSoup(page_source, 'html.parser')
                
                return True
                
            except Exception as e:
                st.error(f'Cannot connect to Chrome driver: {e}')
                return False
            finally:
                if 'driver' in locals():
                    driver.quit()

    def extract_data(self):
        if not self.soup:
            return False
        
        # Extract product names and links
        divs_with_title_links = self.soup.find_all('div', class_='RfADt')
        
        for div in divs_with_title_links:
            try:
                title = div.find('a').get('title')
                href = div.find('a').get('href')
                self.product_name.append(title)
                self.product_links.append(href)
            except AttributeError:
                continue
        
        # Extract prices
        divs_with_price = self.soup.find_all('div', class_='aBrP0')
        
        for div in divs_with_price:
            try:
                price = div.text
                self.prices.append(price)
            except Exception:
                continue
        
        # Extract units sold and locations
        divs_with_units_sold = self.soup.find_all('div', class_='_6uN7R')
        
        for div in divs_with_units_sold:
            try:
                pattern = r'([\d.]+K?) sold'
                match = re.search(pattern, str(div))
                
                if match:
                    number_sold = match.group(1)
                    if 'K' in number_sold:
                        number_sold = int(float(number_sold.replace('K',''))*1000)
                else:
                    number_sold = 0
                
                self.units_sold.append(number_sold)
                
                location_span = div.find('span', class_='oa6ri')
                location = location_span.text.strip() if location_span else "N/A"
                self.locations.append(location)
                
            except Exception:
                continue
        
        return True

    def create_dataframe(self):
        # Ensure all lists have the same length
        min_length = min(len(self.product_name), len(self.prices), 
                        len(self.units_sold), len(self.locations), len(self.product_links))
        
        if min_length == 0:
            return pd.DataFrame()
        
        df = pd.DataFrame({
            'Product Name': self.product_name[:min_length],
            'Price': self.prices[:min_length],
            'Units Sold': self.units_sold[:min_length],
            'Location': self.locations[:min_length],
            'Product Link': self.product_links[:min_length]                           
        })
        
        return df

def scrape_lazada_products(item_name: str, max_pages: int = 3):
    """
    Main function to scrape Lazada products
    """
    config = ScraperConfig(item=item_name, max_pages=max_pages)
    scraper = Scraper(item_name, config)
    all_data = pd.DataFrame()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        for page in range(1, min(max_pages + 1, 6)):  # Limit to max 5 pages
            status_text.text(f'Scraping page {page} of {max_pages}...')
            
            if scraper.scrape_site(page_num=page):
                if scraper.extract_data():
                    page_df = scraper.create_dataframe()
                    if not page_df.empty:
                        all_data = pd.concat([all_data, page_df], ignore_index=True)
                
                # Clear data for next page
                scraper.product_name = []
                scraper.product_links = []
                scraper.prices = []
                scraper.locations = []
                scraper.units_sold = []
            
            progress_bar.progress(page / max_pages)
            
            # Add random delay
            if page < max_pages:
                wait_time = np.random.randint(config.min_random_wait, config.max_random_wait)
                time.sleep(wait_time)
        
        progress_bar.progress(1.0)
        status_text.text('Scraping completed!')
        
        return all_data
        
    except Exception as e:
        st.error(f"Error during scraping: {e}")
        return pd.DataFrame()

def main():
    st.set_page_config(
        page_title="Lazada Product Scraper",
        page_icon="🛍️",
        layout="wide"
    )
    
    st.title("🛍️ Lazada Product Scraper")
    st.markdown("Search and scrape product information from Lazada Singapore")
    
    # Sidebar 
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Item input
        item_name = st.text_input(
            "Enter Product Name:",
            value="Cooling shirt",
            help="Enter the product you want to search for"
        )
        
        # Number of pages
        max_pages = st.slider(
            "Number of pages to scrape:",
            min_value=1,
            max_value=5,
            value=3,
            help="More pages = more data but longer wait time"
        )
        
        # Scrape button
        scrape_button = st.button("🔍 Start Scraping", type="primary")
    
    # Main content area
    if scrape_button and item_name:
        st.markdown(f"### Searching for: **{item_name}**")
        
        with st.spinner("Initializing scraper..."):
            df = scrape_lazada_products(item_name, max_pages)
        
        if not df.empty:
            st.success(f"✅ Successfully scraped {len(df)} products!")
            
            # Display summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Products", len(df))
            
            with col2:
                avg_price = df['Price'].str.replace('S$', '').str.replace(',', '').astype(str)
                numeric_prices = pd.to_numeric(avg_price.str.extract(r'(\d+\.?\d*)')[0], errors='coerce')
                avg_price_val = numeric_prices.mean()
                st.metric("Avg Price (S$)", f"{avg_price_val:.2f}" if not pd.isna(avg_price_val) else "N/A")
            
            with col3:
                total_sold = pd.to_numeric(df['Units Sold'], errors='coerce').sum()
                st.metric("Total Units Sold", f"{total_sold:,.0f}" if not pd.isna(total_sold) else "N/A")
            
            with col4:
                unique_locations = df['Location'].nunique()
                st.metric("Unique Locations", unique_locations)
            
            # Display the data table
            st.markdown("### 📊 Product Data")
                      
  
            filtered_df = df
            
            # Display table with formatting
            st.dataframe(
                filtered_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Product Name": st.column_config.TextColumn(
                        "Product Name",
                        width="large"
                    ),
                    "Price": st.column_config.TextColumn(
                        "Price",
                        width="small"
                    ),
                    "Units Sold": st.column_config.NumberColumn(
                        "Units Sold",
                        format="%d"
                    ),
                    "Location": st.column_config.TextColumn(
                        "Location",
                        width="medium"
                    ),
                    "Product Link": st.column_config.LinkColumn(
                        "Product Link",
                        width="medium"
                    )
                }
            )
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"lazada_{item_name}_products.csv",
                mime="text/csv"
            )
        
        else:
            st.warning("⚠️ No products found. Please try a different search term or check your internet connection.")
    
    elif scrape_button and not item_name:
        st.error("❌ Please enter a product name to search for.")
    
    # Instructions
    with st.expander("📋 Instructions & Requirements"):
        st.markdown("""
        **How to use:**
        1. Enter a product name in the sidebar
        2. Select the number of pages to scrape (1-5)
        3. Click "Start Scraping" to begin
        
        **Requirements:**
        - Chrome browser must be installed
        - ChromeDriver must be installed and accessible via PATH
        - Internet connection required
        
        **Note:**
        - Scraping may take several minutes depending on the number of pages
        - Data can be downloaded as CSV
        """)

if __name__ == "__main__":
    main()
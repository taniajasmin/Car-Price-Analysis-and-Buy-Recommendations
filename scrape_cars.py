import os
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import csv
import schedule
import time
import re
from datetime import datetime

load_dotenv()

def scrape_with_thordata(url):
    headers = {"Authorization": f"Bearer {os.getenv('THORDATA_API_KEY')}", "Content-Type": "application/x-www-form-urlencoded"}
    payload = {"url": url, "type": "html", "js_render": "True"}
    try:
        response = requests.post("https://universalapi.thordata.com/request", data=payload, headers=headers, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return BeautifulSoup(data.get('html', ''), 'html.parser')
        else:
            print(f"Scraping error for {url}: Status {response.status_code}")
            return None
    except Exception as e:
        print(f"Scraping failed for {url}: {str(e)}")
        return None

def scrape_car_listings():
    urls = ["https://www.2dehands.be/autos/?page=1", "https://www.autoscout24.be/nl/aanbod/auto/?page=1&sort=price&desc=0&ustate=N,U"]
    all_cars = []
    
    for url in urls:
        soup = scrape_with_thordata(url)
        if not soup:
            continue
        
        # Try to find any article, div, or li elements that might contain listings
        listings = []
        for tag in ['article', 'div', 'li']:
            potential_listings = soup.find_all(tag, class_=re.compile(r'(listing|item|cldt|hz-Listing)', re.I))
            if potential_listings:
                listings = potential_listings
                print(f"Found {len(listings)} potential listings using <{tag}> on {url}")
                break
        
        if not listings:
            print(f"WARNING: No listings found on {url}")
            continue
        
        for listing in listings[:20]:  # Get more listings to increase chances
            try:
                # Get title
                title_elem = listing.find(['h2', 'h3', 'h4', 'a'])
                title = title_elem.text.strip() if title_elem else ""
                
                # Get URL
                link_elem = listing.find('a', href=True)
                listing_url = ""
                if link_elem:
                    listing_url = link_elem['href']
                    if listing_url.startswith('/'):
                        if 'autoscout' in url:
                            listing_url = f"https://www.autoscout24.be{listing_url}"
                        else:
                            listing_url = f"https://www.2dehands.be{listing_url}"
                    elif not listing_url.startswith('http'):
                        listing_url = ""
                
                # Get all text from the listing for parsing
                raw_text = listing.get_text(separator=' ', strip=True)
                
                # Skip if too little text
                if len(raw_text) < 20:
                    continue
                
                all_cars.append({
                    'title': title,
                    'url': listing_url,
                    'raw_text': raw_text
                })
                    
            except Exception as e:
                print(f"Parsing error: {str(e)}")
                continue
        
        print(f"Collected {len([c for c in all_cars if url.split('/')[2] in c.get('url', '')])} raw listings from {url}")
    
    return all_cars

def parse_car_data(cars):
    """
    Extract: price, year, mileage, brand from raw text
    """
    print("\nParsing car data...")
    
    brands = [
        'Volkswagen', 'VW', 'Mercedes', 'BMW', 'Audi', 'Opel', 'Peugeot', 'Ford',
        'Citroen', 'Renault', 'Toyota', 'Fiat', 'Skoda', 'Nissan', 'Volvo', 'Seat',
        'Hyundai', 'Kia', 'Mazda', 'Honda', 'Suzuki', 'Porsche', 'Land Rover',
        'Lexus', 'Tesla', 'Mini', 'Jeep', 'Alfa Romeo', 'Iveco', 'Dacia'
    ]
    brand_pattern = '|'.join(brands)
    premium_brands = ['BMW', 'Mercedes', 'Audi', 'Porsche', 'Volvo', 'Land Rover', 'Lexus', 'Tesla']
    current_year = datetime.now().year
    
    parsed_cars = []
    
    for car in cars:
        raw_text = car.get('raw_text', '')
        title = car.get('title', '')
        
        # Extract price (€ symbol)
        price_match = re.search(r'€\s*([\d.,]+)', raw_text)
        if price_match:
            price_text = price_match.group(1).replace('.', '').replace(',', '.')
            try:
                price_numeric = float(price_text)
            except:
                price_numeric = 0.0
        else:
            price_numeric = 0.0
        
        # Extract year (4 digits: 19XX or 20XX)
        year_match = re.search(r'(20[0-2]\d|19[89]\d)', raw_text)
        year_numeric = int(year_match.group(1)) if year_match else None
        
        # Extract mileage (numbers followed by km)
        mileage_match = re.search(r'([\d.]+)\s*km', raw_text, re.I)
        if mileage_match:
            try:
                mileage_numeric = int(mileage_match.group(1).replace('.', ''))
            except:
                mileage_numeric = 0
        else:
            mileage_numeric = 0
        
        # Extract brand from title
        brand_match = re.search(f"({brand_pattern})", title, re.IGNORECASE)
        brand = brand_match.group(1) if brand_match else "Unknown"
        
        # Calculate car age
        age = current_year - year_numeric if year_numeric else None
        
        # Mark premium brands
        is_premium = brand.upper() in [b.upper() for b in premium_brands]
        
        # Only keep cars with at least a title and price
        if title and price_numeric > 0:
            parsed_cars.append({
                'title': title,
                'url': car.get('url', ''),
                'price_numeric': price_numeric,
                'year_numeric': year_numeric if year_numeric else 0,
                'mileage_numeric': mileage_numeric,
                'brand': brand,
                'age': age if age else 0,
                'is_premium': is_premium
            })
    
    print(f"\nData extracted:")
    print(f"  Total cars parsed: {len(parsed_cars)}")
    print(f"  Cars with year: {sum(1 for c in parsed_cars if c['year_numeric'] > 0)}")
    print(f"  Cars with mileage: {sum(1 for c in parsed_cars if c['mileage_numeric'] > 0)}")
    print(f"  Cars with brand: {sum(1 for c in parsed_cars if c['brand'] != 'Unknown')}")
    
    return parsed_cars

def save_to_csv(cars):
    headers = ["title", "url", "price_numeric", "year_numeric", "mileage_numeric", "brand", "age", "is_premium"]
    with open('cars_data.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(cars)
    print(f"\nSaved {len(cars)} cars to cars_data.csv")

def scrape_and_save():
    print(f"\n{'='*60}")
    print(f"Starting scrape at {time.ctime()}")
    print(f"{'='*60}")
    
    raw_listings = scrape_car_listings()
    
    if raw_listings:
        parsed_cars = parse_car_data(raw_listings)
        if parsed_cars:
            save_to_csv(parsed_cars)
            print(f"\nScraping completed successfully at {time.ctime()}")
        else:
            print(f"\nNo valid cars found after parsing.")
    else:
        print(f"\nNo listings collected from websites.")

if __name__ == "__main__":
    # Run immediately
    scrape_and_save()
    # Schedule every 15 days
    schedule.every(15).days.do(scrape_and_save)
    while True:
        schedule.run_pending()
        time.sleep(60)
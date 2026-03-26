import json
import time
import urllib.parse
from playwright.sync_api import sync_playwright

def generate_yc_batches():
    """Generates the specific list of batches."""
    seasons = ["Summer", "Spring", "Winter", "Fall"]
    years = range(2026, 2004, -1)
    
    batches = []
    for year in years:
        for season in seasons:
            # Filtering to match your target list logic
            if year == 2005 and season != "Summer": continue
            if year == 2006 and season not in ["Winter", "Summer"]: continue
            batches.append(f"{season} {year}")
    return batches

def scrape_full_batches(output_file="final_yc_links.jsonl"):
    all_links = set()
    batches = generate_yc_batches()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False) # Keep False to monitor
        context = browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        page = context.new_page()

        for batch_name in batches:
            encoded_batch = urllib.parse.quote(batch_name)
            batch_url = f"https://www.ycombinator.com/companies?batch={encoded_batch}"
            print(f"\n--- Processing: {batch_name} ---")
            
            try:
                page.goto(batch_url, timeout=60000)
                
                previous_count = 0
                retries = 0
                while True:
                    # Scroll to the bottom of the current page
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    time.sleep(3) # Give YC a moment to load the next 40
                    
                    # Count how many company links are currently visible
                    current_links = page.query_selector_all('a[href^="/companies/"]')
                    # Filter out non-company links immediately to get an accurate count
                    company_hrefs = [l.get_attribute("href") for l in current_links 
                                    if "/industry" not in l.get_attribute("href")]
                    
                    current_count = len(set(company_hrefs))
                    print(f"[{batch_name}] Found {current_count} links so far...")

                    if current_count == previous_count:
                        retries += 1
                        if retries >= 3: # If count doesn't change after 3 scrolls, batch is done
                            break
                    else:
                        retries = 0 # Reset retries if we found new items
                    
                    previous_count = current_count

                # --- Save Batch Results ---
                batch_added = 0
                for href in set(company_hrefs):
                    full_url = f"https://www.ycombinator.com{href}" if href.startswith("/") else href
                    if full_url not in all_links:
                        all_links.add(full_url)
                        batch_added += 1
                
                print(f"SUCCESS: Added {batch_added} unique links from {batch_name}.")
                print(f"TOTAL COLLECTION: {len(all_links)}")

            except Exception as e:
                print(f"Error on {batch_name}: {e}")

        with open(output_file, 'w') as f:
            for link in all_links:
                f.write(json.dumps({"url": link}) + '\n')
        
        browser.close()

if __name__ == "__main__":
    scrape_full_batches()

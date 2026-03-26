import json
import time
import re
import pandas as pd
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

try:
    from playwright_stealth import stealth_sync
except ImportError:
    from playwright_stealth import stealth as stealth_sync

# CONFIGURATION
SOURCES = [
    {"source": "Forbes_Guide", "url": "https://www.forbes.com/advisor/business/how-to-start-a-business/"},
    {"source": "HK_Companies_Registry", "url": "https://www.cr.gov.hk/en/services/register-company.htm"},
    {"source": "InvestHK_Setup", "url": "https://www.investhk.gov.hk/en/setting-up-in-hong-kong/"},
    {"source": "SME_Link_Resources", "url": "https://www.smelink.gov.hk/en/web/sme-portal/support-for-smes.html"},
    {"source": "BusinessNewsDaily_Retail", "url": "https://www.businessnewsdaily.com/15169-how-to-open-a-retail-store.html"},
    {"source": "YC_Startup_Library", "url": "https://www.ycombinator.com/library"}
]

def clean_educational_content(html):
    """Strips noise and returns clean text for Project 8 analysis."""
    soup = BeautifulSoup(html, "html.parser")
    for noise in soup(["script", "style", "nav", "header", "footer", "form", "aside"]):
        noise.decompose()
    
    main_body = soup.find("article") or soup.find("main") or soup.find("div", {"id": "content"}) or soup.body
    if not main_body: return ""

    text = main_body.get_text(separator="\n")
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def run_scraper(output_file="project8_knowledge_base.jsonl"):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        
        # Use a very specific, modern User-Agent
        context = browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            extra_http_headers={"Referer": "https://www.google.com/"} # Makes it look like you came from a search
        )
        
        page = context.new_page()
        
        try:
            stealth_sync(page) 
        except Exception as e:
            print(f"Stealth could not be applied, proceeding without it: {e}")

        results = []
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in SOURCES:
                print(f"--- Processing: {item['source']} ---")
                try:
                    # 'commit' wait_until is often faster and less likely to time out
                    page.goto(item['url'], wait_until="domcontentloaded", timeout=60000)
                    time.sleep(3) # Short sleep to let JS render final elements

                    clean_text = clean_educational_content(page.content())

                    entry = {
                        "source_name": item['source'],
                        "url": item['url'],
                        "title": page.title(),
                        "content": clean_text,
                        "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    f.write(json.dumps(entry) + '\n')
                    results.append(entry)
                    print(f"SUCCESS: Captured {len(clean_text)} characters.")

                except Exception as e:
                    print(f"FAILED {item['source']}: {str(e)[:60]}...")
                
                time.sleep(2) 

        browser.close()
        return pd.DataFrame(results)

if __name__ == "__main__":
    df = run_scraper()
    if not df.empty:
        print("\nScraping Complete. Knowledge Base Ready for Keyword Extraction.")

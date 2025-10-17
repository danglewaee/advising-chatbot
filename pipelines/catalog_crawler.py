import os, re, time, requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from pathlib import Path

BASE_URL = "https://www.cics.umass.edu/academics/courses"
OUT_DIR = Path("data/raw/catalog")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0"}

def crawl_catalog():
    r = requests.get(BASE_URL, headers=HEADERS)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    links = [urljoin(BASE_URL, a["href"]) for a in soup.select("a[href*='compsci']")]
    links = list(set(links))  # filtering duplicates
    print(f"Found {len(links)} links")

    for link in links:
        try:
            code = re.search(r"compsci[-_]?(\d+)", link, re.I)
            if not code:
                continue
            code = code.group(1)
            res = requests.get(link, headers=HEADERS)
            res.raise_for_status()
            txt = BeautifulSoup(res.text, "html.parser").get_text(separator="\n", strip=True)
            fpath = OUT_DIR / f"compsci{code}.txt"
            fpath.write_text(txt, encoding="utf-8")
            print(f"✅ Saved {fpath}")
            time.sleep(0.5)
        except Exception as e:
            print(f"❌ {link}: {e}")

if __name__ == "__main__":
    crawl_catalog()

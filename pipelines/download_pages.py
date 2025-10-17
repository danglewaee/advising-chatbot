import os
import requests
from bs4 import BeautifulSoup

PAGES = {
    "cics_courses": "https://www.cics.umass.edu/academics/courses?utm_source=chatgpt.com",
    "fall2024-course-description": "https://www.cics.umass.edu/documents/fall-2024-course-descriptions",
    "calendar": "https://www.umass.edu/registrar/academic-calendar",
    "gened": "https://www.umass.edu/registrar/gen-ed-list",
    "cics-academics": "https://www.cics.umass.edu/academics"
}

os.makedirs("data/raw", exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

def clean_html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["header", "footer", "nav", "script", "style"]):
        tag.extract()
    return soup.get_text(separator="\n", strip=True)

def main():
    for name, url in PAGES.items():
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
            text = clean_html_to_text(r.text)
            out_path = os.path.join("data", "raw", f"{name}.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"✅ Saved {out_path} ({len(text)//1000} KB)")
        except Exception as e:
            print(f"❌ {name}: {e}")

if __name__ == "__main__":
    main()

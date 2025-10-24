import os, time, requests
from bs4 import BeautifulSoup
from pathlib import Path
from source_crawler import fetch_and_strip

OUT_DIR = Path("data/raw/bulk")
OUT_DIR.mkdir(parents=True, exist_ok=True)
strip=[5,9]

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

SOURCES = {
    "Fall_2025_Course_Descriptions": "https://content.cs.umass.edu/content/fall-2025-course-description",
    "Spring_2025_Course_Descriptions": "https://content.cs.umass.edu/content/spring-2025-course-descriptions",
    "Fall_2024_Course_Descriptions": "https://content.cs.umass.edu/content/fall-2024-course-descriptions",
    "CICS_Prereq_Changes": "https://www.cics.umass.edu/academics/courses/prerequisite-catalog-and-credit-changes"
}

def main():
    for name, url in SOURCES.items():
        try:
            text = fetch_and_strip(url, strip_from_top=strip[0], strip_from_bottom=strip[1])
            with open(OUT_DIR / f"{name}.txt", "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            print(f"Error occured when trying to save {name}: {e}")

if __name__ == "__main__":
    main()

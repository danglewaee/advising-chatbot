import os, time, requests
from bs4 import BeautifulSoup
from pathlib import Path

OUT_DIR = Path("data/raw/bulk")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

SOURCES = {
    "fall_2025_course_descriptions": "https://content.cs.umass.edu/content/fall-2025-course-description",
    "spring_2025_course_descriptions": "https://content.cs.umass.edu/content/spring-2025-course-descriptions",
    "fall_2024_course_descriptions": "https://content.cs.umass.edu/content/fall-2024-course-descriptions",
    "cics_prereq_changes": "https://www.cics.umass.edu/academics/courses/prerequisite-catalog-and-credit-changes",
    "cs250_syllabus_fall2025": "https://people.cs.umass.edu/~barrington/cs250/fullsyllabus.html",
    "cs250_home_fall2025": "https://people.cs.umass.edu/~barring/cs250/",
}

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["header","footer","nav","script","style"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)

def fetch_and_save(name: str, url: str):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    text = html_to_text(r.text)
    (OUT_DIR / f"{name}.txt").write_text(text, encoding="utf-8")
    print(f"✅ Saved {OUT_DIR / (name + '.txt')} ({len(text)//1000} KB)")

def main():
    for name, url in SOURCES.items():
        try:
            fetch_and_save(name, url)
            time.sleep(0.5)
        except Exception as e:
            print(f"❌ {name}: {e}")

if __name__ == "__main__":
    main()

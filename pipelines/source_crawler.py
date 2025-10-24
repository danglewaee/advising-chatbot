import requests
from bs4 import BeautifulSoup


def fetch_and_strip(url, remove_selectors=None, remove_tag_names=None, strip_from_top=0, strip_from_bottom=0, timeout=20):
    """Fetch a web page and remove likely header/footer elements.

    Strategy:
    - Remove common semantic tags by name (header, footer, nav)
    - Remove elements matching common header/footer selectors (ids/classes)
    - Optionally trim N non-empty lines from top/bottom

    Returns the cleaned text.
    """
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove by tag names first (semantic tags)
    tag_names = remove_tag_names or ["header", "footer", "nav"]
    for tag in tag_names:
        for el in soup.find_all(tag):
            el.decompose()

    # Remove by common header/footer ids/classes
    selectors = remove_selectors or [
        "#header",
        ".header",
        ".site-header",
        "#masthead",
        "#footer",
        ".footer",
        ".site-footer",
    ]
    for sel in selectors:
        for el in soup.select(sel):
            el.decompose()

    text = soup.get_text("\n", strip=True)

    if strip_from_top > 0 or strip_from_bottom > 0:
        lines = [l for l in text.splitlines() if l.strip()]
        if len(lines) > strip_from_top + strip_from_bottom:
            lines = lines[strip_from_top:-strip_from_bottom] if strip_from_bottom > 0 else lines[strip_from_top:]
        text = "\n".join(lines)

    return text


if __name__ == "__main__":
    url = "https://content.cs.umass.edu/content/fall-2025-course-description"
    # tweak strip_lines_from_edges if header/footer are not removed by selectors
    cleaned = fetch_and_strip(url, strip_from_top=5, strip_from_bottom=9)

    with open("test/test_output.txt", "w") as f:
        f.write(cleaned)
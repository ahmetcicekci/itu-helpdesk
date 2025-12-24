import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import json

BASE_URL = "https://yardim.itu.edu.tr/ssslist.aspx"


def fetch_page(url):
    """Download and parse a page into BeautifulSoup."""
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def get_ticket_links(base_url):
    """Return list of all ticket URLs from the list page."""
    soup = fetch_page(base_url)
    panels = soup.find_all("div", class_="panel panel-custom")

    urls = []
    for panel in panels:
        a_tag = panel.find("a", href=True)
        if a_tag:
            urls.append(urljoin(base_url, a_tag["href"]))

    return urls


def get_title(soup):
    h1 = soup.find("h1", class_="text-info")
    if not h1:
        return ""
    return h1.get_text(strip=True)


def get_content(soup):
    panel = soup.find("div", class_="panel-body")
    if not panel:
        return ""

    h1 = panel.find("h1", class_="text-info")
    if h1:
        h1.extract()

    return panel.get_text("\n", strip=True)


def get_unit(soup):
    footer = soup.find("div", class_="panel-footer")
    if not footer:
        return ""
    a_tag = footer.find("a")
    if not a_tag:
        return ""
    return a_tag.get_text(strip=True)


def get_labels(soup):
    tag_div = soup.find("div", id="etiket")
    if not tag_div:
        return []

    links = tag_div.find_all("a", class_="label label-custom")
    return [t.get_text(strip=True) for t in links]


def get_categories(soup):
    cat_div = soup.find("div", id="kategori")
    if not cat_div:
        return []

    cat_list_div = cat_div.find("div", class_="well list")
    if not cat_list_div:
        return []

    categories = []
    items = cat_list_div.find_all("li")

    for li in items:
        a_tag = li.find("a", href=True)
        if a_tag:
            if a_tag.has_attr("title"):
                categories.append(a_tag["title"].strip())
            else:
                categories.append(a_tag.get_text(strip=True))

    return categories



def scrape_tickets(limit=0, output_file="itu_help_data.jsonl"):
    ticket_urls = get_ticket_links(BASE_URL)
    print(f"Found {len(ticket_urls)} ticket URLs.")

    if limit > 0: 
        ticket_urls = ticket_urls[:limit]
    results = []

    for url in ticket_urls:
        print("Processing:", url)

        try:
            soup = fetch_page(url)

            title = get_title(soup)
            content = get_content(soup)
            unit = get_unit(soup)
            categories = get_categories(soup)
            labels = get_labels(soup)

            results.append({
                "url": url,
                "title": title,
                "unit": unit,
                "categories": categories,
                "labels": labels,
                "content": content
            })

        except Exception as e:
            print("Error:", e)
            results.append({
                "url": url,
                "title": "",
                "unit": "",
                "categories": [],
                "labels": [],
                "content": ""
            })

    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")

    print(f"\nSaved {len(results)} records to {output_file}")


if __name__ == "__main__":
    scrape_tickets()

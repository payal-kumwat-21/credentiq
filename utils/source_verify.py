import requests

NEWS_API_KEY = "b11ee384cad34d448f0d5f663b9dd63e"

TRUSTED_DOMAINS = [
    "bbc.co.uk",
    "reuters.com",
    "thehindu.com",
    "timesofindia.indiatimes.com",
    "pib.gov.in"
]

def verify_sources(query):
    url = "https://newsapi.org/v2/everything"

    params = {
        "q": query,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 5,
        "apiKey": NEWS_API_KEY
    }

    response = requests.get(url, params=params)
    data = response.json()

    matches = []

    if "articles" in data:
        for article in data["articles"]:
            source_url = article.get("url", "")

            for domain in TRUSTED_DOMAINS:
                if domain in source_url:
                    matches.append({
                        "title": article.get("title"),
                        "source": article.get("source", {}).get("name"),
                        "url": source_url
                    })

    return matches

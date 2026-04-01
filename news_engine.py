import requests

API_KEY = "b11ee384cad34d448f0d5f663b9dd63e"



def get_news(category="general"):
    url = f"https://newsapi.org/v2/top-headlines?country=us&category={category}&apiKey={API_KEY}"

    try:
        res = requests.get(url).json()
        return res.get("articles", [])
    except:
        return []

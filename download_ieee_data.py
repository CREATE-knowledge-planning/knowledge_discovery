import json

import requests


def download_ieee_tgrs_paper_info():
    start_record = 1
    payload = {
        'apikey': 'ey7d7xf7y7nz9b5e8e8msbdr',
        'format': 'json',
        'max_records': 200,
        'start_record': start_record,
        'sort_order': 'desc',
        'sort_field': 'article_number',
        'publication_title': 'IEEE Transactions on Geoscience and Remote Sensing'
    }
    articles = []
    for i in range(100):
        print(start_record)
        r = requests.get("http://ieeexploreapi.ieee.org/api/v1/search/articles", params=payload)
        papers_json = r.json()
        if 'articles' in papers_json:
            articles.extend(papers_json['articles'])
        start_record += 200
        payload['start_record'] = start_record
    with open('papers_test.json', 'w', encoding='utf-8') as papers_json_file:
        json.dump(articles, papers_json_file)


if __name__ == '__main__':
    download_ieee_tgrs_paper_info()

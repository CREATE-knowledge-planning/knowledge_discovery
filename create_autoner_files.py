import json


def create_dict_core():
    pass


def create_dict_full():
    pass


def create_raw_text():
    with open('papers_test.json') as papers_json_file:
        papers_list = json.load(papers_json_file)
    with open('raw_text.txt', 'w', encoding='utf-8') as raw_text_file:
        for paper in papers_list:
            for word in paper["title"].split():
                raw_text_file.write(word + '\n')
            raw_text_file.write('\n')
            if "abstract" in paper:
                for word in paper["abstract"].split():
                    raw_text_file.write(word + '\n')
                raw_text_file.write('\n')


if __name__ == '__main__':
    create_raw_text()
    create_dict_core()
    create_dict_full()

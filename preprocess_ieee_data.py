import json
import spacy

nlp = spacy.load("en_core_web_sm")

with open('./papers_test.json', encoding='utf-8') as papers_json_file, open('./papers_test.txt', 'w', encoding='utf-8') as papers_txt_file:
    papers_list = json.load(papers_json_file)
    for paper in papers_list:
        papers_txt_file.write(paper["title"] + '\n')
        if "abstract" in paper:
            abstract_doc = nlp(paper["abstract"])
            for sentence in abstract_doc.sents:
                papers_txt_file.write(sentence.text + '\n')

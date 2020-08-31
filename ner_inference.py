import json

from lightner import decoder_wrapper
import spacy

nlp = spacy.load("en_core_web_sm")

if __name__ == '__main__':
    model = decoder_wrapper(".\\AutoNER\\models\\BC5CDR\\checkpoint\\autoner\\best.th")
    with open('papers_test.json') as papers_json_file:
        papers_list = json.load(papers_json_file)
    with open('inference_results.txt', 'w', encoding='utf-8') as inference_results_file:
        for paper in papers_list:
            title_doc = nlp(paper["title"])
            token_list = [token.text for token in title_doc]
            result = model.decode(token_list)
            inference_results_file.write(result + '\n')
            if "abstract" in paper:
                abstract_doc = nlp(paper["abstract"])
                for sentence in abstract_doc.sents:
                    spacy_sen = nlp(sentence.text)
                    token_list = [token.text for token in spacy_sen]
                    result = model.decode(token_list)
                    inference_results_file.write(result + '\n')

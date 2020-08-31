import json

from gensim.corpora import WikiCorpus
from gensim.corpora.textcorpus import TextCorpus
from gensim import utils
import gensim.models
import spacy

nlp = spacy.load("en_core_web_sm")


class CorpusTGRS(TextCorpus):
    def getstream(self):
        num_texts = 0
        with open(self.input, encoding='utf-8') as f:
            for line in f:
                yield line
                num_texts += 1

        self.length = num_texts

    def get_texts(self):
        for doc in self.getstream():
            spacy_doc = nlp(doc)
            yield [token.lower_ for token in spacy_doc]


class TGRSSentences:
    def __init__(self, tgrss_path, wikipeida_path):
        self.corpus1 = CorpusTGRS(tgrss_path)
        self.corpus2 = WikiCorpus(wikipeida_path)


    def __iter__(self):
        for sentence in self.corpus1.get_texts():
            yield list(sentence)
        for sentence in self.corpus2.get_texts():
            yield list(sentence)


if __name__ == '__main__':
    # Load ieee tgrs
    tgrs_sentences = TGRSSentences('./papers_test.txt', './enwiki-latest-pages-articles.xml.bz2')
    print("Sentences loaded")

    model = gensim.models.Word2Vec(sentences=tgrs_sentences, size=200, workers=16)
    model.save('tgrs_embeddings.model')

from collections import defaultdict
import logging

from gensim import corpora, models, similarities

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]

def load_documents(documents):
    stoplist = set('for a of the and to in'.split())
    texts = [[word for word in document.lower().split() if word not in stoplist]
             for document in documents]

    return texts

def find_best_answer(texts, question):
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    model = models.TfidfModel(corpus, normalize=True)

    vec_bow = dictionary.doc2bow(question.lower().split())
    vec_tfidf = model[vec_bow]

    index = similarities.MatrixSimilarity(model[corpus])

    sims = index[vec_tfidf]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])

    return sims[0][0]

question = "survey"

texts = load_documents(documents)
answer_index = find_best_answer(texts, question)

print(documents[answer_index])

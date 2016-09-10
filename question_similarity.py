from collections import defaultdict
import logging

from gensim import corpora, models, similarities

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def parse_interview(filename):
    question_answers = {}
    current_question = ""
    with open(filename) as f:
        for line in f:
            if line.startswith("##"):
                current_question = line[2:].strip()
            elif line.startswith("#"):
                if current_question not in question_answers:
                    question_answers[current_question] = ''
                    question_answers[current_question] += line[1:].strip()
    return question_answers

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

def get_trump_answer(question):
    question_answers = parse_interview("interviews.txt")
    documents = list(question_answers.keys())
    texts = load_documents(question_answers.keys())

    answer_index = find_best_answer(texts, question)

    return question_answers[documents[answer_index]]

print(get_trump_answer('PennApps'))

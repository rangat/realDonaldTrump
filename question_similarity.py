from collections import defaultdict
import logging, os, re

from gensim import corpora, models, similarities

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def parse_interview(speeches_dir):
    question_answers = {}
    current_question = ""
    for filename in os.listdir(speeches_dir):
        filename = os.path.join(speeches_dir, filename)
        with open(filename) as f:
            print(filename)
            for line in f:
                if line.startswith("Q:"):
                    current_question = line[2:].strip()
                elif line.startswith("TRUMP:"):
                    if current_question not in question_answers:
                        question_answers[current_question] = ''
                        question_answers[current_question] += line[6:].strip()
    return question_answers

def load_documents(documents):
    stoplist = []

    with open("stopwords.txt") as f:
        for line in f:
            stoplist.append(line.strip())
    stoplist = set(stoplist)

    texts = []
    for document in documents:
        text = []
        for word in document.lower().split():
            # Strip contractions
            word = re.sub(r'[\'’][a-z]{1,2}', "", word)
            # Strip grammatical markings
            word = re.sub(u'[,?.;\'-._!—&]', "", word)
            if word and word not in stoplist:
                text.append(word)
        if text:
            texts.append(text)

    return texts

def find_best_answer(texts, question):
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    model = models.TfidfModel(corpus, normalize=True)

    # LSI experimentation
    corpus_tfidf = model[corpus]
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)
    corpus_lsi = lsi[corpus_tfidf]
    lsi.print_topics(2)

    vec_bow = dictionary.doc2bow(question.lower().split())
    vec_tfidf = model[vec_bow]

    index = similarities.MatrixSimilarity(model[corpus])

    sims = index[vec_tfidf]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])

    return sims[0][0]

def get_trump_answer(question):
    question_answers = parse_interview('speeches')
    documents = list(question_answers.keys())
    texts = load_documents(question_answers.keys())

    answer_index = find_best_answer(texts, question)

    return question_answers[documents[answer_index]]

print(get_trump_answer('Thank you'))

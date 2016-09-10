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
    keyword_full_answer_mapping = {}
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

            # Generate the key from the keywords
            text.sort()
            key = "".join(text)
            keyword_full_answer_mapping[key] = document

    return keyword_full_answer_mapping, texts

def find_best_answer(texts, processed_question):
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    tfidf = models.TfidfModel(corpus, normalize=True)

    corpus_tfidf = tfidf[corpus]
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=300)
    corpus_lsi = lsi[corpus_tfidf]

    vec_bow = dictionary.doc2bow(processed_question)
    vec_lsi = lsi[vec_bow]

    index = similarities.MatrixSimilarity(lsi[corpus])

    sims = index[vec_lsi]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    return sims[0][0]

def get_trump_answer(question):
    question_answers = parse_interview('speeches')
    keyword_full_answer_mapping, texts = load_documents(question_answers.values())

    _, processed_question = load_documents([question])

    answer_index = find_best_answer(texts, processed_question[0])
    key = "".join(texts[answer_index])

    return keyword_full_answer_mapping[key]

print(get_trump_answer('What is the best thing about Obama'))

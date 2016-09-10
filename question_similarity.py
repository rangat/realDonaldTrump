import logging
import os
import json
import re

from gensim import corpora, models, similarities
import markovify
import nltk
from nltk.tag.perceptron import PerceptronTagger

tagger = PerceptronTagger()
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

read_from_cache = False


class POSifiedText(markovify.NewlineText):
    def word_split(self, sentence):
        tagset = None
        tokens = nltk.word_tokenize(sentence)
        tags = nltk.tag._pos_tag(tokens, tagset, tagger)
        words = ["::".join(tag) for tag in tags]
        return words

    def word_join(self, words):
        sentence = " ".join(word.split("::")[0] for word in words)
        return sentence


def parse_corpus(identifier, speeches_dir):
    answers = []
    for filename in os.listdir(speeches_dir):
        filename = os.path.join(speeches_dir, filename)
        with open(filename) as f:
            for line in f:
                if line.startswith(identifier):
                    identifier_length = len(identifier)
                    answers.append(line[identifier_length:].strip())
    return answers


def extract_keywords(documents):
    stoplist = []
    stemmer = nltk.stem.SnowballStemmer('english')

    with open("stopwords.txt") as f:
        for line in f:
            word_stem = stemmer.stem(line.strip())
            stoplist.append(word_stem)
    stoplist = set(stoplist)

    texts = []
    keyword_full_answer_mapping = {}
    for document in documents:
        text = []
        for word in document.lower().split():
            # Strip contractions
            word = re.sub(r'[\'’][a-z]{1,2}', "", word)
            # Strip grammatical markings
            word = ''.join(e for e in word if e.isalpha())

            word_stem = stemmer.stem(word)

            if word_stem and word_stem not in stoplist:
                text.append(word_stem)
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

    question_vector_bow = dictionary.doc2bow(processed_question)
    question_vector_lsi = lsi[question_vector_bow]

    index = similarities.MatrixSimilarity(lsi[corpus])

    simularity_scores = index[question_vector_lsi]
    simularity_scores = sorted(enumerate(simularity_scores), key=lambda item: -item[1])

    return simularity_scores


def get_answer(identifier, directory, question):
    if read_from_cache:
        keyword_full_answer_mapping = json.load(open(os.path.join(directory, "keyword_full_answer_mapping.json"), 'r'))
        texts = json.load(open(os.path.join(directory, "texts.json"), 'r'))
    else:
        corpus = parse_corpus('{identifier}:'.format(identifier=identifier), directory)
        keyword_full_answer_mapping, texts = extract_keywords(corpus)

        json.dump(keyword_full_answer_mapping, open(os.path.join(directory, "keyword_full_answer_mapping.json"),'w'))
        json.dump(texts, open(os.path.join(directory, "texts.json"),'w'))

    _, processed_question = extract_keywords([question])

    if processed_question:
        similarity_scores = find_best_answer(texts, processed_question[0])
        if similarity_scores[0][1] > 0:
            answer_index = similarity_scores[0][0]
            key = "".join(texts[answer_index])
            return keyword_full_answer_mapping[key]

    return generate_sentences(keyword_full_answer_mapping)


def generate_sentences(keyword_full_answer_mapping):
    markov_text = []
    for answer in keyword_full_answer_mapping.values():
        markov_text.append(answer)

    model_all = POSifiedText("\n".join(markov_text))

    generated_sentence = None
    while generated_sentence is None:
        generated_sentence = model_all.make_short_sentence(140)
    generated_sentence = " ".join(word.split("::")[0] for word in generated_sentence.split(" "))
    return generated_sentence

print(get_answer('TRUMP', 'trump_speeches', 'what\'s going on?'))

from collections import defaultdict
import logging, os, re

from gensim import corpora, models, similarities
import markovify
import nltk

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class POSifiedText(markovify.NewlineText):
    def word_split(self, sentence):
        words = re.split(self.word_split_pattern, sentence)
        words = [ "::".join(tag) for tag in nltk.pos_tag(words) ]
        return words

    def word_join(self, words):
        sentence = " ".join(word.split("::")[0] for word in words)
        return sentence

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

    return sims

def get_trump_answer(question):
    question_answers = parse_interview('speeches')
    keyword_full_answer_mapping, texts = load_documents(question_answers.values())

    _, processed_question = load_documents([question])

    if processed_question:
        similarity_scores = find_best_answer(texts, processed_question[0])
        if similarity_scores[0][1] > 0:
            answer_index = similarity_scores[0][0]
            key = "".join(texts[answer_index])
            return keyword_full_answer_mapping[key]

    return generate_trump_sentences(keyword_full_answer_mapping)

def generate_trump_sentences(keyword_full_answer_mapping):
    markov_text = []
    for answer in keyword_full_answer_mapping.values():
        markov_text.append(answer)

    model_all = POSifiedText("\n".join(markov_text))

    with open("speeches/famous_quotes.txt") as f:
        quotes = f.read()

    model_quotes = POSifiedText(quotes)
    model_combo = markovify.combine([ model_quotes, model_all ], [ 3, 1 ])

    generated_sentence = None
    while generated_sentence is None:
        generated_sentence = model_combo.make_short_sentence(140)
    generated_sentence = " ".join(word.split("::")[0] for word in generated_sentence.split(" "))
    return generated_sentence

print(get_trump_answer('how are you'))

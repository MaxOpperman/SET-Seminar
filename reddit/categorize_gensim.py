import os
import re

import pickle

import pandas as pd
from gensim.utils import simple_preprocess
import spacy
import nltk
import gensim.corpora as corpora
from gensim import models
from gensim.models import CoherenceModel
from tqdm import tqdm

from nltk.corpus import stopwords


def sent_to_words(sentences):
    for sentence in sentences:
        yield simple_preprocess(str(sentence), deacc=True)  # deacc=True removes punctuations


def remove_stopwords(texts, stop_words):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts, trigram_mod, bigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def lemmatization(texts, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def compute_coherence_values(dictionary, corpus, texts, limit, start=1, step=1):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in tqdm(range(start, limit, step), desc="Computing coherence values"):
        model = models.LdaMulticore(corpus=corpus, num_topics=num_topics, id2word=dictionary)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in tqdm(enumerate(ldamodel[corpus]), desc="Format topics sentences"):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(
                    pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True
                )
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return sent_topics_df


def categorize_data(prepared_data, pickle_folder):
    corpus = pickle.load(open(f"./pickle/{pickle_folder}/corpus", 'rb'))  # list of text/doc
    model = pickle.load(open(f"./pickle/{pickle_folder}/optimal_lda_model", 'rb'))  # LDA model

    df_topic_sents_keywords = format_topics_sentences(ldamodel=model, corpus=corpus, texts=prepared_data)
    with open(f"redditManualKeywords.csv", "w+", newline="\n", encoding="utf-8") as keywords_file:
        df_topic_sents_keywords.to_csv(keywords_file)
    return (df_topic_sents_keywords['Dominant_Topic'],
            df_topic_sents_keywords['Perc_Contribution'],
            df_topic_sents_keywords['Topic_Keywords'])


def classify_comments(comments: [str], pickle_folder, num_topics, train_set_size) -> [str]:
    # Remove punctuation and convert to lowercase
    comments_processed = [(re.sub('[,/.!?:]', '', str(x))).lower() for x in comments]
    data = comments_processed.copy()

    if not os.path.exists(f'./pickle/{pickle_folder}'):
        os.makedirs(f'./pickle/{pickle_folder}')

    data_words = list(sent_to_words(data))
    if train_set_size is not None:
        data_words = list(sent_to_words(data[:min(len(data) - 1, train_set_size)]))

    # Build the bigram and trigram models
    bigram = models.Phrases(data_words, min_count=5, threshold=100)
    # higher threshold fewer phrases
    trigram = models.Phrases(bigram[data_words], threshold=100)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = models.phrases.Phraser(bigram)
    trigram_mod = models.phrases.Phraser(trigram)
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    nlp.max_length = 3000000

    # install sentiment Natural Language Toolkit package
    try:
        nltk.find('stopwords')
    except LookupError:
        nltk.download('stopwords')

    # Remove stop words
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'ie', 'st', 'th', 'rd', 'copyright'])
    data_words_nostops = remove_stopwords(data_words, stop_words)
    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops, bigram_mod)
    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # pickle dump data
    pickle.dump(data_words, open(f"./pickle/{pickle_folder}/data_words", "wb"))
    pickle.dump(data_lemmatized, open(f"./pickle/{pickle_folder}/data_lemmatized", "wb"))

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)
    # Create Corpus
    texts = data_lemmatized
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    pickle.dump(id2word, open(f"./pickle/{pickle_folder}/id2word", "wb"))
    pickle.dump(corpus, open(f"./pickle/{pickle_folder}/corpus", "wb"))

    if num_topics is None:
        num_topics = 3
        # start, limit, step = 2, 3, 1
        # x = range(start, limit, step)
        # model_list, coherence_values = compute_coherence_values(
        #     dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=start, limit=limit, step=step)
        # best_result_index = coherence_values.index(max(coherence_values))
        # print(f"The {x[best_result_index]} topics gives the highest coherence score of "
        #       f"{coherence_values[best_result_index]}")
        # num_topics = x[best_result_index]
    # Final Model
    print("Generating model...")
    final_model = models.LdaMulticore(corpus=corpus,
                                      id2word=id2word,
                                      num_topics=num_topics,
                                      random_state=100,
                                      chunksize=100,
                                      passes=10)
    print("Model generated...")
    pickle.dump(final_model, open(f"./pickle/{pickle_folder}/optimal_lda_model", "wb"))

    return categorize_data(comments_processed, pickle_folder)

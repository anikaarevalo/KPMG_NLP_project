import nltk
import re
import spacy
import gensim
from typing import List

from gensim.utils import simple_preprocess

stop_words = nltk.corpus.stopwords.words('dutch')

nlp = spacy.load('nl_core_news_sm', disable=['parser', 'ner'])

empty_words = [
    'advies',
    'algemeen',
    'art',
    'artikel',
    'bedoelen',
    'bepalen',
    'bepaling',
    'besluit',
    'betreffen',
    'bevoegd',
    'deel',
    'drie',
    'een',
    'één',
    'gegeven',
    'gelet',
    'geven',
    'goed',
    'hoofdstuk',
    'inlichting',
    'lid',
    'leten',
    'maatregel',
    'minister',
    'naam',
    'nemen',
    'ontwerp',
    'overwegen',
    'overwegende',
    'paragraaf',
    'persoon',
    'toepassing',
    'twee',
    'volgen',
    'voorzien',
    'wet'
]


def initial_clean(text:str) ->  str:
    '''
    Function that cleans articles and tries add spaces where they are missing

    :params text: str that is the text

    return a str
    '''
    pattern = r'(,|:|;)'
    pat = re.compile(pattern)
    new_text = pat.sub(r'\1  ', text)

    pattern = r'([\s][\w-]+\.)([A-Za-z]+)'
    pat = re.compile(pattern)
    new_text = pat.sub(r'\1  \2', new_text)

    pattern = r'([\"][\.])'
    pat = re.compile(pattern)
    new_text = pat.sub(r'\1 ', new_text)

    pattern = r'(__+)'
    pat = re.compile(pattern)
    new_text = pat.sub(r' \1  ', new_text)

    new_text = re.sub(r" +"," ",new_text)

    pattern = r'([A-Za-z]+)([A-Z][a-z])'
    pat = re.compile(pattern)
    new_text = pat.sub(r'\1 \2', new_text)

    return new_text


def sent_to_words(sentences: List[str]):
    '''
    Generator that when iterated over preprocesses for each text in the given list

    :param sentences: list of str that represent legal texts
    '''
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations




def bigrammodel(data_words: List[List[str]]) -> gensim.models.phrases.Phraser:
    '''
    Function that returns a bigram model for a given corpus
    
    :param: list of list of str that represent words of texts
    '''
    # Building the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) 
    bigram_mod = gensim.models.phrases.Phraser(bigram)

    return bigram_mod





def remove_stopwords(texts: List[List[str]]) -> List[List[str]]:
    '''
    Function that return a given texts without stop words

    :param: List of strings that are texts
    '''
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts : List[List[str]], bigram_mod: gensim.models.phrases.Phraser) -> List[List[str]]:
    '''
    Function that finds bigrams in list of list of words
    
    :param texts: list of list of words that represent a list of texts
    :param bigram_mod: the model needed to create bigrams
    '''
    return [bigram_mod[doc] for doc in texts]

def lemmatization(texts : List[List[str]], allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']) -> List[List[str]]:
    '''
    Function that lemmatizes all words in a list of texts

    :param texts: list of list of words representing list of texts
    :param allowed_postags: list of postags that we allow to passed on the return statement
    '''
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if (token.pos_ in allowed_postags) and (token.lemma_ not in  empty_words )])
    return texts_out


def preproces(texts: List[str]) -> List[List[str]]:
    '''
    function that lemmatizes a body of texts and returns the lemma of each word

    :param texts: list of str representing list of legal texts:
    '''

    # initial cleaning of the text
    data = [initial_clean(text) for text in texts]

    # Creating the list of words
    data_words = list(sent_to_words(data))

    # Creating the bigram model
    bigram_mod = bigrammodel(data_words)

    # Removing Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Forming Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops, bigram_mod)

    # Doing lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    return data_lemmatized
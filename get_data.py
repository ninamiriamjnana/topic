from peeweemodels import *

import logging

import itertools

import gensim

import nltk
from nltk.collocations import TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures

import re

#for creating the frequency dictionary
from collections import defaultdict

# set of stopwords for german
stopset = set(nltk.corpus.stopwords.words('german'))


logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO  # ipython sometimes messes up the logging setup; restore

def iterate_over_posts():
    mysql_db.connect()
    posts=Post.select()
    for post in posts.naive().iterator():
        yield post.text

def iter(): # print only the first 3 posts -> these three posts are "gone" after the iterator went over!
    post_stream=iterate_over_posts()
    print(list(itertools.islice(post_stream, 1)))

    print(list(itertools.islice(post_stream, 1)))


# now a new class so that we can iterate several times, without losing the items

class PPosts(object):
    
    def __iter__(self):
        posts=Post.select()
        for post in posts.naive().iterator():
            # tokenize each message; simply lowercase & match alphabetic chars, for now
            yield list(gensim.utils.tokenize(post.text, lower=True))



# print the first two tokenized messages
#print(list(itertools.islice(tokenized_corpus, 2)))

def best_ngrams(words, top_n=10, min_freq=5):
    """
    Extract `top_n` most salient collocations (bigrams and trigrams),
    from a stream of words. Ignore collocations with frequency
    lower than `min_freq`.

    This fnc uses NLTK for the collocation detection itself -- not very scalable!

    Return the detected ngrams as compiled regular expressions, for their faster
    detection later on.

    """
    tcf = TrigramCollocationFinder.from_words(words)
    tcf.apply_freq_filter(min_freq)
    trigrams = [' '.join(w) for w in tcf.nbest(TrigramAssocMeasures.chi_sq, top_n)]
    logging.info("%i trigrams found: %s..." % (len(trigrams), trigrams[:20]))

    bcf = tcf.bigram_finder()
    bcf.apply_freq_filter(min_freq)
    bigrams = [' '.join(w) for w in bcf.nbest(BigramAssocMeasures.pmi, top_n)]
    logging.info("%i bigrams found: %s..." % (len(bigrams), bigrams[:20]))

    pat_gram2 = re.compile('(%s)' % '|'.join(bigrams), re.UNICODE)
    pat_gram3 = re.compile('(%s)' % '|'.join(trigrams), re.UNICODE)

    print pat_gram2
    
    return pat_gram2, pat_gram3


class PPosts_Collocations(object): 
    """ damit mach ich tokens wo die haeufigsten tri- und bigrams gleich
    zusammengenommen werden"""
    def __init__(self):
        logging.info("collecting ngrams from postd")
        # generator of documents; one element = list of words
        posts=Post.select()
        documents=(self.split_words(post.text) for post in posts.naive().iterator())
        # generator: concatenate (chain) all words into a single sequence, lazily
        words = itertools.chain.from_iterable(documents)
        self.bigrams, self.trigrams = best_ngrams(words)



    def split_words(self, text, stopwords=stopset): # here: no lemmatization!
        """
        Break text into a list of single words. Ignore any token that falls into
        the `stopwords` set.

        """
        return [word
                for word in gensim.utils.tokenize(text, lower=True)
                if word not in stopset and len(word) > 3] # stopset from pattern

    def tokenize(self, message):
        """
        Break text (string) into a list of Unicode tokens.
        
        The resulting tokens can be longer phrases (collocations) too,
        e.g. `new_york`, `real_estate` etc.

        """
        text = u' '.join(self.split_words(message))
        text = re.sub(self.trigrams, lambda match: match.group(0).replace(u' ', u'_'), text)
        text = re.sub(self.bigrams, lambda match: match.group(0).replace(u' ', u'_'), text)
        return text.split()

    def __iter__(self):
        posts=Post.select()
        for post in posts.naive().iterator():
            yield self.tokenize(post.text)

    """   

    frequency = defaultdict(int)
    for text in texts: # hier muss ich ja eigentlich das __iter__ aufrufen!
        for token in text: # das hier habe ich ja schon? ich krieg ja immer einen token zurueck mit dem iter
           frequency[token] += 1

    from collections import defaultdict
    fq= defaultdict( int )
    for w in words:
    fq[w] += 1

    a = [1,1,1,1,2,2,2,2,3,3,4,5,5]
    >>> d = {x:a.count(x) for x in a}
    >>> d
    {1: 4, 2: 4, 3: 2, 4: 1, 5: 2}"""

""" ok das versteh ich nicht. was ich jetzt machen muss: ein dictionary. mit was? wozu?"""
""" DAS BRAUCH ICH: The dictionary object now contains all words that appeared in the corpus, along with how many times they appeared. """
#%time collocations_corpus = PPosts_Collocation
# doc_stream = (tokens for _, tokens in collocations_corpus)
# id2word_wiki = gensim.corpora.Dictionary(doc_stream)
#print(list(itertools.islice(collocations_corpus, 2)))"""

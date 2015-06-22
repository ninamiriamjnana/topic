from peeweemodels import *

import logging

import itertools

import gensim

from gensim import corpora

import nltk
from nltk.collocations import TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures

import re

#for creating the frequency dictionary
from collections import defaultdict

# set of stopwords for german
#stopset = set(nltk.corpus.stopwords.words('german'))
stopset=set(line.strip() for line in open('stopwords_de.txt'))

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
        documents=(self.split_words(post.text, stopset) for post in posts.naive().iterator())
        # generator: concatenate (chain) all words into a single sequence, lazily
        words = itertools.chain.from_iterable(documents)
        self.bigrams, self.trigrams = best_ngrams(words)



    def split_words(self, text, stopset): # here: no lemmatization!
        """
        Break text into a list of single words. Ignore any token that falls into
        the `stopwords` set.

        """
        return [word
                for word in gensim.utils.tokenize(text, lower=True)
                if word not in stopset and len(word) > 3] # stopset from pattern
        """        
        return [word
                for word in gensim.utils.tokenize(text, lower=True)
                if word not in stopset and len(word) > 3] # stopset from pattern
        """
    def tokenize(self, message):
        """
        Break text (string) into a list of Unicode tokens.
        
        The resulting tokens can be longer phrases (collocations) too,
        e.g. `new_york`, `real_estate` etc.

        """
        text = u' '.join(self.split_words(message, stopset))
        text = re.sub(self.trigrams, lambda match: match.group(0).replace(u' ', u'_'), text)
        text = re.sub(self.bigrams, lambda match: match.group(0).replace(u' ', u'_'), text)
        return text.split()

    def __iter__(self):
        posts=Post.select()
        for post in posts.naive().iterator():
            yield self.tokenize(post.text)


    


def make_vector():
 
    corpus=PPosts_Collocations()
    frequency = defaultdict(int)  
          
    for post in corpus: # hier muss ich ja eigentlich das __iter__ aufrufen!
        for token in post: # das hier habe ich ja schon? ich krieg ja immer einen token zurueck mit dem iter
            frequency[token] += 1

    from pprint import pprint   # pretty-printer
    pprint(frequency) #frequency dict

    pdict = corpora.Dictionary(corpus) # dict with ids DAS IST DAS ID 2 WORD WIKI

    pprint(pdict)

    print(pdict.token2id)

    return pdict
 
class PPostCorpus(object):
    def __init__(self,dictionary):
        """
        Parse the first `clip_docs` Wikipedia documents from file `dump_file`.
        Yield each document in turn, as a list of tokens (unicode strings).
        
        """
        self.dictionary = dictionary
        logging.info("collecting ngrams from postd")
        # generator of documents; one element = list of words
        posts=Post.select()
        documents=(self.split_words(post.text,stopset) for post in posts.naive().iterator())
        # generator: concatenate (chain) all words into a single sequence, lazily
        words = itertools.chain.from_iterable(documents)
        self.bigrams, self.trigrams = best_ngrams(words)

    def split_words(self, text, stopset): # here: no lemmatization!
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
        text = u' '.join(self.split_words(message,stopset))
        text = re.sub(self.trigrams, lambda match: match.group(0).replace(u' ', u'_'), text)
        text = re.sub(self.bigrams, lambda match: match.group(0).replace(u' ', u'_'), text)
        return text.split()
        

    def __iter__(self):
      posts=Post.select()
      for post in posts.naive().iterator():
          yield self.dictionary.doc2bow(self.tokenize(post.text))
    
"""   

# create a stream of bag-of-words vectors
id2word=make_vector()
#pcorpus = PPostCorpus(id2word) JAHAAAA
# %time gensim.corpora.MmCorpus.serialize('premium_bow.mm', pcorpus) store vector
#mm_corpus = gensim.corpora.MmCorpus('premium_bow.mm')
#print(mm_corpus)
%time lda_model = gensim.models.LdaModel(mm_corpus, num_topics=20, id2word=id2word, passes=50)
_ = lda_model.print_topics(-1)

Transformation can be stacked. For example, here we'll train a TFIDF model, and then train Latent Semantic Analysis on top of TFIDF:
%time tfidf_model = gensim.models.TfidfModel(mm_corpus, id2word=id2word)
The TFIDF transformation only modifies feature weights of each word. Its input and output dimensionality are identical (=the dictionary size).

# cache the transformed corpora to disk, for use in later notebooks
%time gensim.corpora.MmCorpus.serialize('./data/wiki_tfidf.mm', tfidf_model[mm_corpus])
%time gensim.corpora.MmCorpus.serialize('./data/wiki_lsa.mm', lsi_model[tfidf_model[mm_corpus]])

tfidf_corpus = gensim.corpora.MmCorpus('./data/wiki_tfidf.mm')
# `tfidf_corpus` is now exactly the same as `tfidf_model[wiki_corpus]`
print(tfidf_corpus)

lsi_corpus = gensim.corpora.MmCorpus('./data/wiki_lsa.mm')
# and `lsi_corpus` now equals `lsi_model[tfidf_model[wiki_corpus]]` = `lsi_model[tfidf_corpus]`
print(lsi_corpus)

# store all trained models to disk
lda_model.save('./data/lda_wiki.model')
lsi_model.save('./data/lsi_wiki.model')
tfidf_model.save('./data/tfidf_wiki.model')
id2word_wiki.save('./data/wiki.dictionary')

# load the same model back; the result is equal to `lda_model`
same_lda_model = gensim.models.LdaModel.load('./data/lda_wiki.model')

# select top 50 words for each of the 20 LDA topics
top_words = [[word for _, word in lda_model.show_topic(topicno, topn=50)] for topicno in range(lda_model.num_topics)]
print(top_words)

# get all top 50 words in all 20 topics, as one large set
all_words = set(itertools.chain.from_iterable(top_words))

for topicno, words in enumerate(top_words):
    print("%i: %s" % (topicno, ' '.join(words[:10])))

write to file:
f = open('topics','w')
for topicno, words in enumerate(top_words):
    s=u"%i: %s \n" % (topicno, u' '.join(words[:10]))
    s.encode('utf8')
    f.write(s)  
f.close  
f.write('hi there\n') # python will convert \n to os.linesep
f.close()
"""



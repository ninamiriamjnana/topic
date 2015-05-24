from peeweemodels import *

import logging

import itertools

import gensim


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

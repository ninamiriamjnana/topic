from peeweemodels import *

def iterate_over_posts():
    mysql_db.connect()
    posts=Post.select()
    for post in posts.naive().iterator():
        print post.text

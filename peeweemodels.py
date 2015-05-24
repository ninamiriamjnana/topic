from peewee import *

mysql_db= MySQLDatabase('arachnophobia', host='localhost', user='nina', passwd='pw' )

class MySQLModel (Model):
	class Meta:
        	database = mysql_db

# http://peewee.readthedocs.org/en/latest/peewee/api.html#CompositeKey



class User (MySQLModel):
	name = CharField(index=True)

class Topic (MySQLModel):
	name = CharField(index=True)

class Post (MySQLModel):
    pid=IntegerField(index=True)
    text=CharField()
    timestamp=CharField()
    order=IntegerField()
    user= ForeignKeyField (User)	
    topic= ForeignKeyField (Topic)

class HTMLPage(MySQLModel):
      url=CharField(index=True)
      body=TextField ()


def create_tables():	
    User.create_table()
    Topic.create_table()    
    Post.create_table() 
    HTMLPage.create_table()   
	

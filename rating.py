'''ORM for rating table'''

# pylint: disable-msg=E1101,R0903
from sqlalchemy import Table
from sqlalchemy.orm import mapper
import db

rating_table = Table('rating', db.meta, autoload=True)

class Rating(object):
	'''User rating of a movie'''
	def __init__(self, movie, rating):
		self.movie = movie
		self.rating = rating

mapper(Rating, rating_table)

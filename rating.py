from sqlalchemy import Table
from sqlalchemy.orm import mapper, join
import db

rating_table = Table('rating', db.meta, autoload=True)

class Rating(object):
	def __init__(self, movie, rating):
		self.movie = movie
		self.rating = rating

mapper(Rating, rating_table)

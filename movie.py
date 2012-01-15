from sqlalchemy import Table
from sqlalchemy.orm import mapper, join, relation, backref
import db
import rating

movie_table = Table('movie', db.meta, autoload=True)
movie_search_table = Table('movie_search', db.meta, autoload=True)

class Movie(object):
	'''Movie info - name & year'''

	def __init__(self, name, year):
		self.name = name
		self.year = year

	@classmethod
	def search(self, session, name):
		'''Search for a movie'''
		j = join(movie_search_table, movie_table, movie_search_table.c.movie_id == Movie.movie_id)
		q = session.query(Movie).select_from(j)
		return q.filter(movie_search_table.c.name.match(name))

mapper(Movie, movie_table, properties = {
	'rating' : relation(rating.Rating, uselist=False, backref="movie")
})


if __name__ == '__main__':
	session = db.Session()

	for row in Movie.search(session, 'story'):
		print row.year, row.name

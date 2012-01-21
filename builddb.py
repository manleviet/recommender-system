'''Standalone script to populate the database from movies.txt.'''
import sqlite3
import codecs
import re

conn = sqlite3.connect('movies.sqlite')
insert = 'insert into movie(name, year) values(?, ?)'

pattern = re.compile('^(.*)\s\((\d\d\d\d)\)\s*(\(V\))?$')

movies = []
with codecs.open('movie_ids.txt', encoding='latin1') as movie_file:
	with conn:
		i = 0
		for line in movie_file:
			print i
			i += 1
			movie = line.lstrip('01234567890').strip()

			match = pattern.match(movie)
			if match:
				movie = match.group(1)
				year = match.group(2)
			else:
				year = None
			conn.execute(insert, (movie, year))

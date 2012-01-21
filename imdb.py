'''Standalone script to load imdb ratings from a CSV file into the database'''
import db
import csv
import movie
import rating

session = db.Session()

def choose(movies, name, year):
	'''Pick the right movie from the search results'''
	while len(movies) > 1:
		print 'Multiple results for %s (%s):' % (name, year)
		for i, m in enumerate(movies):
			print '\t%d) %s (%s)' % (i + 1, m.name, m.year)
		print 'Choose an option, or "skip" to skip.'
		choice = raw_input('> ')
		if choice.lower() == 'skip':
			return None
		try:
			m = movies[int(choice) - 1]
		except:
			print 'Invalid choice'
		else:
			return m

	if movies:
		return movies[0]
	else:
		return None

with open('ratings.csv', 'r') as f:
	f.next() # skip header
	reader = csv.reader(f)
	for row in reader:
		name = unicode(row[5], encoding='utf-8')
		imdb_rating = int(row[8])
		year = int(row[11])

		movies = movie.Movie.search(session, name).all()

		# Make sure it's the right year
		#movies = filter(lambda m: m.year == year, movies)

		# Handle multiple results for the same movie
		m = choose(movies, name, year)

		if m:
			print 'Adding %s' % (name,)
			r = rating.Rating(m, imdb_rating)
			session.add(r)
		else:
			print 'Skipping %s (not found in database)' % (name,)

session.commit()

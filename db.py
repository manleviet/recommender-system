'''Set up SqlAlchemy for the sqlite database.'''

from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker

engine = create_engine('sqlite:///movies.sqlite', echo=False)
meta = MetaData(bind=engine)
Session = sessionmaker(bind=engine)

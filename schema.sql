-- Movie info
create table movie (
	movie_id integer constraint movie_pk primary key,
	name text,
	year integer
);

-- User ratings
create table rating (
	rating_id integer constraint rating_pk primary key,
	movie_id integer constraint rating_movie_id_fk references movie(movie_id) on delete cascade constraint rating_movie_id_u unique,
	rating integer
);

-- Stored feature data
create table movie_feature (
	movie_feature_id integer constraint movie_feature_pk primary key,
	movie_id integer constraint movie_feature_movie_id_fk references movie(movie_id) on delete cascade,
	feature_id integer,
	value float
);

-- User weights
create table preference (
	feature_id integer constraint preference_pk primary key,
	weight float
);

-- Recommendations
create view recommendation as
	select
		movie.name,
		movie.year,
		sum(weight * value) as score
	from
		movie
		inner join movie_feature on (movie.movie_id = movie_feature.movie_id)
		inner join preference on (preference.feature_id = movie_feature.feature_id)
	where movie.movie_id not in (select movie_id from rating)
	group by movie.movie_id
	order by score desc
;

-- Search
create virtual table movie_search using fts3 (
	name,
	movie_id integer constraint movie_search_movie_id_fk references movie(movie_id)
);

create trigger populate_movie_search after insert on movie
begin
	insert into movie_search(name, movie_id) values(new.name, new.movie_id);
end;

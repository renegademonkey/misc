"""

00_task_instructions.py

Description

In this exercise you will need to develop your own recommendation system using information about :

·  users
·  movies
·  preferences of some users towards some movies

Using your recommendation system you will need to predict preferences of users towards movies for new (user,movie) pairs.



For this exercise, you’ll only need to download these files :
·  movies_metadata.csv : contains the movies
·  ratings.csv : contains the movie ratings for all the users
·  evaluation_ratings.csv : the pairs of (user, movie) for which a rating is expected in the submission file

Evaluation

Your solution will be evaluated using root mean square prediction error :


f(i) is your forecast for i-th (user,movie) pair rating
y(i) is the true rating for i-th (user,movie) pair
N - total number of (user,movie) pairs in the test set



Submission Format

1.       The source code must be submitted (preferably github)
2.       The submission file : for the (user,movie) pairs contained in evaluation_ratings.csv, this file should contain three columns: UserId, MovieId and Rating.

Bonus

a bonus submission file : for all the movies that are rated in movies_metadata.csv file, this file should contain pairs of (movie, movie) (movies are distinct) containing three columns: Id, MovieId, MovieId and Rating

Evaluation criteria

The test will be evaluated based upon these criteria :

1.       code quality
2.       machine learning usage (which method, data cleaning, calibration, ...)
3.       tests
4.       industrialization level (build/run)
5.       documentation (how to build/run...)

"""
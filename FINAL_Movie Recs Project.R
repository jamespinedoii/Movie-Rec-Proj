if(!require(tidyverse)) install.packages("tidyverse", 
                                         repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", 
                                     repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", 
                                          repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()

download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)

colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, 
                                  times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#Creating a test and train set from edx
#Test set will be 10% of edx
set.seed(1, sample.kind="Rounding")
test_index_edx <- createDataPartition(y = edx$rating, 
                                      times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index_edx,]
test_set <- edx[test_index_edx,]

# Make sure userId and movieId in test set are also in train set
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

#Access year data for each film in the train set
if(!require(dplyr)) install.packages("dplyr", 
                                         repos = "http://cran.us.r-project.org")
if(!require(tidyr)) install.packages("tidyr", 
                                     repos = "http://cran.us.r-project.org")
if(!require(stringr)) install.packages("stringr", 
                                     repos = "http://cran.us.r-project.org")
library(dplyr)
library(tidyr)
library(stringr)

movie_info <- train_set$title

years <- sub("\\).*", "", sub(".*\\(", "", movie_info)) 

train_set <- train_set %>% mutate(year = as.numeric(years))

# Adding year data to test set
movie_info_test <- test_set$title

years_test <- sub("\\).*", "", sub(".*\\(", "", movie_info_test)) 

test_set <- test_set %>% mutate(year = as.numeric(years_test))

#setting a baseline
mu_hat <- mean(train_set$rating)
mu_hat

naive_rmse <- RMSE(test_set$rating, mu_hat)
naive_rmse

#use year of movie to predict rating
#older movies have higher ratings - the classic effect
mu <- mean(train_set$rating) 
classic_movie_avgs <- train_set %>% 
  group_by(year) %>% 
  summarize(c_e = mean(rating - mu))

predicted_ratings <- test_set %>% 
  left_join(classic_movie_avgs, by='year') %>%
  mutate(pred = mu + c_e) %>%
  pull(pred)

#Calc RMSE with Classic Effect
RMSE(predicted_ratings, test_set$rating)

#A little help - but not much

#Genre Effect

genre_effect <- train_set %>% 
  left_join(classic_movie_avgs, by='year') %>%
  group_by(genres) %>% 
  summarize(g_e = mean(rating - mu - c_e))

predicted_ratings <- test_set %>% 
  left_join(classic_movie_avgs, by='year') %>%
  left_join(genre_effect, by='genres') %>%
  mutate(pred = mu + c_e + g_e) %>%
  pull(pred)

#Calc RMSE with Classic Effect and Genre Effect
RMSE(predicted_ratings, test_set$rating)

#Calculating User Bias
user_avgs <- train_set %>%
  left_join(classic_movie_avgs, by='year') %>%
  left_join(genre_effect, by='genres') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - c_e - g_e))

predicted_ratings <- test_set %>% 
  left_join(classic_movie_avgs, by='year') %>%
  left_join(genre_effect, by='genres') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + c_e + g_e + b_u)  %>%
  pull(pred)

#Calc RMSE with Classic Effect, Genre Effect and User Bias
RMSE(predicted_ratings, test_set$rating)

#Cacl Rating Factor - some movies have high or low ratings
rating_avgs <- train_set %>% 
  left_join(classic_movie_avgs, by='year') %>%
  left_join(genre_effect, by='genres') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(movieId) %>% 
  summarize(r_f = mean(rating - mu - c_e - g_e - b_u))

predicted_ratings <- test_set %>% 
  left_join(classic_movie_avgs, by='year') %>%
  left_join(genre_effect, by='genres') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(rating_avgs, by='movieId') %>%
  mutate(pred = mu + c_e + g_e + b_u + r_f)

#Calc RMSE with Classic Effect, Genre Effect, User Bias and Rating Factor
RMSE(predicted_ratings$pred, test_set$rating)

#This gives us a RMSE that is acceptable, but can we do better?

#Will try to regularize all of our effects, shown by adding "_r"

lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(train_set$rating)
  
  r_f_r <- train_set %>% 
    group_by(movieId) %>%
    summarize(r_f_r = sum(rating - mu)/(n()+l))
  
  b_u_r <- train_set %>% 
    left_join(r_f_r, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u_r = sum(rating - r_f_r - mu)/(n()+l))
  
  g_e_r <- train_set %>% 
    left_join(r_f_r, by="movieId") %>%
    left_join(b_u_r, by="userId") %>%
    group_by(genres) %>%
    summarize(g_e_r = sum(rating - r_f_r - b_u_r - mu)/(n()+l))
  
  c_e_r <- train_set %>% 
    left_join(r_f_r, by="movieId") %>%
    left_join(b_u_r, by="userId") %>%
    left_join(g_e_r, by="genres") %>%
    group_by(year) %>%
    summarize(c_e_r = sum(rating - r_f_r - b_u_r - g_e_r - mu)/(n()+l))
  
  predicted_ratings <- 
    test_set %>% 
    left_join(r_f_r, by = "movieId") %>%
    left_join(b_u_r, by = "userId") %>%
    left_join(g_e_r, by="genres") %>%
    left_join(c_e_r, by="year") %>%
    mutate(pred = mu + r_f_r + b_u_r + g_e_r + c_e_r) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, test_set$rating))
})

lambda <- lambdas[which.min(rmses)]
lambda

min(rmses)

#RMSE is much better!

#Breaking the best lambda out of the function above
#And running this algorithm on edx in prep for testing on validation

#First need to add years to edx

movie_info_edx <- edx$title

years_edx <- sub("\\).*", "", sub(".*\\(", "", movie_info_edx)) 

edx <- edx %>% mutate(year = as.numeric(years_edx))

#now running lambda on edx

r_f_r <- edx %>% 
  group_by(movieId) %>%
  summarize(r_f_r = sum(rating - mu)/(n()+lambda))

b_u_r <- edx %>% 
  left_join(r_f_r, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u_r = sum(rating - r_f_r - mu)/(n()+lambda))

g_e_r <- edx %>% 
  left_join(r_f_r, by="movieId") %>%
  left_join(b_u_r, by="userId") %>%
  group_by(genres) %>%
  summarize(g_e_r = sum(rating - r_f_r - b_u_r - mu)/(n()+lambda))

c_e_r <- edx %>% 
  left_join(r_f_r, by="movieId") %>%
  left_join(b_u_r, by="userId") %>%
  left_join(g_e_r, by="genres") %>%
  group_by(year) %>%
  summarize(c_e_r = sum(rating - r_f_r - b_u_r - g_e_r - mu)/(n()+lambda)) 

#Adding Years Column to Validation Set
movie_info_validation <- validation$title

years_validation <- sub("\\).*", "", sub(".*\\(", "", movie_info_validation)) 

validation <- validation %>% mutate(year = as.numeric(years_validation))

#time to run this on validation set to calc final RMSE

predicted_ratings_final <- 
  validation %>% 
  left_join(r_f_r, by = "movieId") %>%
  left_join(b_u_r, by = "userId") %>%
  left_join(g_e_r, by="genres") %>%
  left_join(c_e_r, by="year") %>%
  mutate(pred = mu + r_f_r + b_u_r + g_e_r + c_e_r) %>%
  pull(pred)

RMSE(predicted_ratings_final, validation$rating)

#this reports an RMSE of 0.8648142 - good for this project's purposes!
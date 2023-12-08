# Steam Game Recommender
This project focuses on building recommender systems to make predictions related to video game reviews from the Steam platform. 
Two main prediction tasks are addressed: play prediction and time played prediction.
Overview
Play Prediction
The play prediction task involves predicting, for a given (user, game) pair from 'pairs Played.csv', whether the user would play the game (0 or 1). 
The accuracy of the model will be measured in terms of categorization accuracy, representing the fraction of correct predictions. 
The test set has been carefully constructed to maintain a balance, with exactly 50% of pairs corresponding to played games and the other 50% not played.

Time Played Prediction
For (user, game) pairs in 'pairs Hours.csv', the project aims to predict how long a person will play a game. 
The time played is transformed as log2(hours + 1). The accuracy of this prediction will be measured using mean-squared error (MSE).

Data
The data used for training and evaluation are sourced from the Steam platform. 
The 'pairs Played.csv' dataset contains information about whether a user has played a particular game, while 'pairs Hours.csv' provides details about the time played.

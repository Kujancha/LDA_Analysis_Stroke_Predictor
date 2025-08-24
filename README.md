# Stroke Predictor using the Linear Discriminatory Analysis
This little thing uses information about a patient's health status, gender, lifestyle to predict a stroke, although after training multiple times, it does have a somewhat significant False Negatives, I tried many train-test splits but the amount of False Negatives is dispropotionaley high. Maybe, a Logisitc Regression will give better results.

The LDA Model used here was built from stracth in Python using the Generative Model. There are two ways of modelling a Linear Discriminatory Analysis, the other being Fisher's LDA which uses the concept of projection and EigenVectors are also used in the latter method. The former one is derived from the Baye's Formula, but both methods involve a lot of matrix operations.

Thought the high False Negative was due to my perhaps inferior implementatiom but Sklearn's LDA also showed the same reuslts.

The test data was taken from Kaggle. 
https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
This data about Strokes (Brain Stroke ?, Heart Stroke?, i don't know lol) to be exact.

## Also hate that the code to clean the .csv file took more line of codes that the actual LDA impelementation

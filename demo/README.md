# Demo to demonstrate how ml-model works
Model learned to detect genres of given dialogue. Demo is implemented as simple web service with one input window for dialogue considered and one button to make prediction.

## Repository content

* templates - folder with html template of web page
* classifier.py - module provides class Classifier which can load model and make prediction 
* dialogs_vectorizer.pkl - dump of learned tfidf vectorizer 
* dialogue_example_good_will_hunting.txt - dialogue example for testing service
* model_dump.pkl - dump of learned OneVsRestClassifier with LogisticRegression as estimator
* movie_genres_baseline_2.ipynb - jupyter notebook with model
* threshold.csv - best thresholds of probability for each genre for classify dealogue
* web.py - flask app module

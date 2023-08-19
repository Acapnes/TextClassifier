import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, classification_report

# Read the necessary files for create education set
sport = pd.read_csv('./sport.txt', header=None, names=['text'], sep='\t', quoting=3)
economy = pd.read_csv('./economy.txt', header=None, names=['text'], sep='\t', quoting=3)
technology = pd.read_csv('./technology.txt', header=None, names=['text'], sep='\t', quoting=3)

# Create education set
train_set = pd.concat([sport, economy, technology])
# train_set['label'] = ['sport'] * len(sport) + ['economy'] * len(economy) + ['technology'] * len(technology)
train_set['label'] = ['sport'] * len(sport) + ['economy'] * len(economy) + ['technology'] * len(technology)

# Read test set
test_set = pd.read_csv('./test.txt', header=None, names=['text'], sep='\t', quoting=3)

# Create Naive Bayes classifier and educate
nb_pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

nb_pipeline.fit(train_set['text'], train_set['label'])
nb_predictions = nb_pipeline.predict(test_set['text'])

# Classify by using SVM and Education set
svm_pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', LinearSVC())
])

svm_pipeline.fit(train_set['text'], train_set['label'])
svm_predictions = svm_pipeline.predict(test_set['text'])

# Compate classifiers performance 

test_set['label'] = nb_predictions

# Measure Naive Bayes performance
print('Naive Bayes:')
print(nb_predictions)
print(confusion_matrix(test_set['label'], nb_predictions))
print(classification_report(test_set['label'], nb_predictions, zero_division=True))

# Measure SVM'nin performance
print('SVM:')
svm_predictions = svm_pipeline.predict(test_set['text'])
print(svm_predictions)
print(confusion_matrix(test_set['label'], svm_predictions))
print(classification_report(test_set['label'], svm_predictions, zero_division=True))
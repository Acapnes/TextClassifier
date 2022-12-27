import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, classification_report

# Eğitim setini oluşturmak için gerekli olan dosyaları oku
spor = pd.read_csv('./spor.txt', header=None, names=['text'], sep='\t', quoting=3)
ekonomi = pd.read_csv('./ekonomi.txt', header=None, names=['text'], sep='\t', quoting=3)
teknoloji = pd.read_csv('./teknoloji.txt', header=None, names=['text'], sep='\t', quoting=3)

# Eğitim setini oluştur
train_set = pd.concat([spor, ekonomi, teknoloji])
# train_set['label'] = ['spor'] * len(spor) + ['ekonomi'] * len(ekonomi) + ['teknoloji'] * len(teknoloji)
train_set['label'] = ['spor'] * len(spor) + ['ekonomi'] * len(ekonomi) + ['teknoloji'] * len(teknoloji)

# Test setini oku
test_set = pd.read_csv('./test.txt', header=None, names=['text'], sep='\t', quoting=3)

# Naive Bayes sınıflandırıcısını oluştur ve eğit
nb_pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

nb_pipeline.fit(train_set['text'], train_set['label'])
nb_predictions = nb_pipeline.predict(test_set['text'])

# SVM sınıflandırıcısını kullanarak eğitim setini kullanarak sınıflandırma yapın
svm_pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', LinearSVC())
])

svm_pipeline.fit(train_set['text'], train_set['label'])
svm_predictions = svm_pipeline.predict(test_set['text'])

# Sınıflandırıcıların performansını karşılaştırın

test_set['label'] = nb_predictions

# Naive Bayes'in performansını ölçün
print('Naive Bayes:')
print(nb_predictions)
print(confusion_matrix(test_set['label'], nb_predictions))
print(classification_report(test_set['label'], nb_predictions, zero_division=True))

# SVM'nin performansını ölçün
print('SVM:')
svm_predictions = svm_pipeline.predict(test_set['text'])
print(svm_predictions)
print(confusion_matrix(test_set['label'], svm_predictions))
print(classification_report(test_set['label'], svm_predictions, zero_division=True))
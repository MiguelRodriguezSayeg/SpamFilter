import pandas as pd
import numpy as np
import seaborn as sn
import itertools
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn import metrics
from roc import ROC_Curve

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

'''emails' csv downloaded from Kaggle's spam dataset https://www.kaggle.com/karthickveerakumar/spam-filter'''
    
data = pd.read_csv("YOUR_PATH\\emails.csv",encoding='latin-1')

out = data['spam']
text = data['text']

label=np.array(out)
text=np.array(text)


x_train = []
x_test = []
y_train = []
y_test = []

text, label = shuffle(text,label)
x_train, x_test, y_train, y_test = train_test_split(text,label,train_size=0.8,test_size=0.2)


x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)



count_vect = CountVectorizer(decode_error='ignore')
x_train_count = count_vect.fit_transform(x_train)
tfidf_trans = TfidfTransformer()
x_train_tfidf = tfidf_trans.fit_transform(x_train_count)

x_test_count = count_vect.transform(x_test)
x_test_tfidf = tfidf_trans.transform(x_test_count)

model1 = GaussianNB()
model2 = LinearSVC()

model1.fit(x_train_tfidf.toarray(),y_train)
model2.fit(x_train_tfidf,y_train)
y_pred1 = model1.predict(x_test_tfidf.toarray())
y_pred2 = model2.predict(x_test_tfidf)

print(accuracy_score(y_test,y_pred1))
print(accuracy_score(y_test,y_pred2))



cnf_matrix = metrics.confusion_matrix(y_test, y_pred1)
cnf_matrix2 = metrics.confusion_matrix(y_test, y_pred2)
np.set_printoptions(precision=2)


plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[0,1], normalize=True,
                      title='Normalized confusion matrix for Gaussian NB')
plt.figure()
plot_confusion_matrix(cnf_matrix2, classes=[0,1], normalize=True,
                      title='Normalized confusion matrix for SVC')


plt.show()



testVar = input("Write your e-mail.")
sample_text="Subject: "+testVar
sample_text=[sample_text]
x_sample_count = count_vect.transform(sample_text)
x_sample_tfidf = tfidf_trans.transform(x_sample_count)


if (model1.predict(x_sample_tfidf.toarray())==[1]):
    print("The Gaussian NB model thinks the email is spam.")
else:
    print("The Gaussian NB model thinks the email is not spam.")

if(model2.predict(x_sample_tfidf)==[1]):
    print("The SVC model thinks the email is spam.")
else:
    print("The SVC model thinks the email is not spam.")

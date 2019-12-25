from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

dat = pd.read_csv('yeast.csv')

cols = ['mcg', 'gvh', 'alm', 'mit', 'erl', 'pox', 'vac', 'nuc']
x = dat[cols]  # Feature apa yang akan menjadi parameter klasifikasi
y = dat['class']  # Label kelas hasil klasifikasi

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1)  # Test_size 20% dan training set 80%

# Membuat classifier
naive = GaussianNB()
# Melatih classifier dengan menggunakan data training
naive = naive.fit(x_train, y_train)

# Melakukan prediksi terhadap data testing beserta probabilitasnya
y_predict = naive.predict(x_test)
naive.predict_proba(x_test)

print(classification_report(y_test, y_predict))
print(confusion_matrix(y_test, y_predict))

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Untuk me-load dataset ke dalam Pandas Dataframe
dat = pd.read_csv('yeast.csv')

cols = ['mcg', 'gvh', 'alm', 'mit', 'erl', 'pox', 'vac', 'nuc']
x = dat[cols]  # Feature apa yang akan menjadi parameter klasifikasi
y = dat['class']  # Label kelas hasil klasifikasi


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1)  # Test_size 20% dan training set 80%

# menyetarakan data
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# x_train = sc.fit_transform(x_train)
# x_test = sc.transform(x_test)

print(x_train)

# Membuat classifier
model = SVC(kernel='rbf')
# model = DecisionTreeClassifier()

# Melatih classifier dengan menggunakan data training
model = model.fit(x_train, y_train)

# Melakukan prediksi terhadap data testing beserta probabilitasnya
y_predict = model.predict(x_test)
# model.predict_proba(x_test)

print(classification_report(y_test, y_predict))
print(confusion_matrix(y_test, y_predict))

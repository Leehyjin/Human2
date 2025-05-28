from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv('./dataset/sms_spam.csv', encoding='ISO-8859-1')

data = data[['v1', 'v2']] 
data.columns = ['label', 'message']
X=data['message']
y=data['label']

#데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train) 
X_test = vectorizer.transform(X_test) 


knn= KNeighborsClassifier(n_neighbors=3) #k=3 가장 가까운 3개의 데이터를 참고하겠다. 
knn.fit(X_train, y_train) 

#테스트 데이터 예측
y_pred=knn.predict(X_test)

#모델 성능 평가
print("모델 성능 평가: ")
print("정확도: ", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
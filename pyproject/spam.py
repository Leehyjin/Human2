import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix
import seaborn as sns
#from ydata_profiling import ProfileReport


data = pd.read_csv('./dataset/sms_spam.csv', encoding='ISO-8859-1')

data.drop_duplicates(inplace=True)

#프로파일링 리포트 생성
#profile = ProfileReport(data, title="SMS Spam Datasets", explorative=True)

#리포트 저장 및 출력
#profile.to_file("sms_spam_profiling_report.html")


data = data[['v1', 'v2']].copy()
data.columns = ['label', 'message']

# 데이터 전처리
data.loc[:,'label']=data['label'].map({'ham': 0, 'spam': 1})
data.dropna(inplace=True) #NaN 값이 있는 행 제거, 원본 자체 수정
data['label'] = data['label'].astype(int)

#학습 및 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

#TF-IDF 벡터화: Term Frequency: 문장 안에서 단어가 얼마나 자주 등장하는지/IDF: INverse Document Frequency

vectorizer=TfidfVectorizer(stop_words='english')
X_train_tfidf=vectorizer.fit_transform(X_train)
X_test_tfidf=vectorizer.transform(X_test)


#모델학습
model = MultinomialNB()
model.fit(X_train_tfidf, y_train) #백터화된 요소와 타겟 값 훈련

#예측
y_pred=model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("=\nClassification Report:\n", classification_report(y_test, y_pred))

#샘플 테스트 
sample_text =["Congratulations! You've been selected to win a brand new iPhone.",
"Click the link now to claim your prize: www.fake-prize.com."]
sample_tfidf = vectorizer.transform(sample_text)
predictions = model.predict(sample_tfidf)

for text, label in zip(sample_text, predictions):
    print(f"Text: {text} => {'Spam' if label ==1 else 'Ham'}")


# Streamlit UI
st.title("SMS Spam Classifier") 
st.write("영문 메세지를 입력하면, 스팸인지 아닌지 분류해줍니다.")

#사용자 입력 받기
user_input = st.text_area("메세지를 입력해주세요", height=150)
                          

#버튼 누르면 예측 수행 
if st.button("Check Message"):
    messages =[line.strip() for line in user_input.split('\n') if line.strip() != ""]
       
    if not messages: 
        st.warning("메세지를 입력해주세요.")
    else:
        input_tfidf = vectorizer.transform(messages)
        predictions = model.predict(input_tfidf)    
        
        #결과 출력 
        st.markdown("### 스팸 분류 결과")
        for msg, pred in zip(messages, predictions):
            label = "Ham(Not spam)" if pred==0 else "spam"
            st.markdown(f"**Message:** {msg}  \n**Prediction:** {label}")
    
    
    st.write("")
    st.markdown("### **워드 클라우드**")

# 시각화 
# 1. 워드 클라우드 
    user_text =' '.join(messages)
    wc = WordCloud(width=800, height=400, background_color='white').generate(user_text)
    st.image(wc.to_array())
    
    st.write("")
    st.write("")
    st.write("")
    st.markdown("### **1. 모델 성능**")

    # 2. 모델 성능 시각화
    cm = confusion_matrix(y_test, y_pred) #y_test: 스팸 여부 실제값, y_pred: 스팸 여부 예측값
    fig, ax=plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Ham', 'Spam'],
                yticklabels=['Ham', 'Spam'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual') 
    st.pyplot(fig)
   
  # 3. 메세지 길이에 따른 스팸 여부 시각화(실제값)
    st.markdown("### **2. 메세지 길이에 따른 스팸 여부**")
    
    # 메세지 길이 열 추가
    data['s'] = data['message'].apply(len)
  
    #히스토그램
    fig2, ax2 = plt.subplots()
    sns.histplot(data=data, x='message_length', hue='label', bins=30, palette={0: 'green', 1: 'red'}, kde=True, ax=ax2)
    ax2.set_xlabel('Message Length')
    ax2.set_ylabel('Count') 
    ax2.legend(title='Label', labels=['Ham', 'Spam'])
    st.pyplot(fig2)
    

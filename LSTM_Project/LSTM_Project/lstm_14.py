import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import streamlit as st

plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 기본 한글 폰트

# 파일 경로
file_path = "전라남도 나주시_시내버스 노선별 승하차 인원_20250601.csv"
df = pd.read_csv(file_path, encoding='euc-kr')

# 중복행 제거 및 결측값 제거
df = df.drop_duplicates()
df = df.dropna()

stop_id_to_name = df[['정류소ID', '정류소명']].drop_duplicates().set_index('정류소ID')['정류소명'].to_dict()
stop_name_to_id = {v: k for k, v in stop_id_to_name.items()}

# 날짜와 요일 처리
df['날짜'] = pd.to_datetime(df['날짜'])
df['요일'] = df['날짜'].dt.dayofweek

# 정류소별 일별 승차인원 집계
daily_df = df.groupby(['정류소ID', '날짜', '요일'])['승차인원'].sum().reset_index()
daily_df = daily_df.sort_values(['정류소ID', '날짜'])

# 요일 원-핫 인코딩
weekday_dummies = pd.get_dummies(daily_df['요일'], prefix='요일')
daily_df = pd.concat([daily_df, weekday_dummies], axis=1)

# 시퀀스 생성
seq_length = 14
X_data = {}
y_data = {}
scalers = {}

for stop_id, group in daily_df.groupby('정류소ID'):
    group = group.sort_values('날짜')
    feature_cols = ['승차인원'] + list(weekday_dummies.columns)
    features = group[feature_cols].values

    # 승차인원 정규화
    scaler = MinMaxScaler()
    features[:, 0:1] = scaler.fit_transform(features[:, 0:1])
    scalers[stop_id] = scaler

    X_seq = []
    y_seq = []

    for i in range(len(features) - seq_length):
        seq_x = features[i:i+seq_length]
        last_weekday = np.argmax(seq_x[-1][1:])  # 마지막 날 요일

        for j in range(i + seq_length, len(features)):
            future_weekday = np.argmax(features[j][1:])
            if future_weekday == last_weekday:
                seq_y = features[j][0]  # 해당 요일의 승차인원
                X_seq.append(seq_x)
                y_seq.append(seq_y)
                break

    if X_seq:
        X_data[stop_id] = np.array(X_seq, dtype=np.float32)
        y_data[stop_id] = np.array(y_seq, dtype=np.float32)

# 선택 옵션 구성
stop_keys = list(X_data.keys())
stop_name_options = [f"{stop_id_to_name[stop_id]} (ID: {stop_id})" for stop_id in stop_keys]
selected_option = st.sidebar.selectbox("정류장 선택", stop_name_options)

# ID 추출
try:
    selected_stop_id = float(selected_option.split("(ID: ")[1].rstrip(")"))
except (AttributeError, IndexError, ValueError):
    st.error("정류장 ID 파싱 중 오류가 발생했습니다. 정류장을 다시 선택해주세요.")
    st.stop()
selected_stop_name = stop_id_to_name[selected_stop_id]

# X/y 안전하게 불러오기
if selected_stop_id not in X_data:
    st.error("선택한 정류소에 대한 학습 데이터가 부족하여 예측이 불가능합니다.")
    st.stop()

X = X_data[selected_stop_id]
y = y_data[selected_stop_id]

# 학습 / 검증 분할
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 모델 구성
model = Sequential([
    LSTM(64, input_shape=(X.shape[1], X.shape[2])),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# 모델 학습
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# 모델 학습 시각화
plt.figure(figsize=(12,6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 예측
y_pred = model.predict(X_test)

# 정규화 복원
scaler = scalers[selected_stop_id]
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# 평가지표 계산
y_true = y_test_inv.flatten()
y_pred_flat = y_pred_inv.flatten()
mae = mean_absolute_error(y_true, y_pred_flat)
mse = mean_squared_error(y_true, y_pred_flat)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred_flat)

# Streamlit UI
st.title("나주시 정류장 승차인원 예측 결과 비교")

test_dates = (
    daily_df[daily_df['정류소ID'] == selected_stop_id]
    .iloc[-len(y_test):]['날짜']
    .dt.strftime('%Y-%m-%d')
    .tolist()
)

compare_df = pd.DataFrame({
    '날짜': test_dates,
    '실제 승차인원': y_test_inv.flatten(),
    '예측 승차인원': y_pred_inv.flatten()
})
compare_df['날짜'] = pd.to_datetime(compare_df['날짜']).dt.strftime('%Y-%m-%d')

st.sidebar.markdown(f"### 선택된 정류소: {selected_stop_name} (ID: {selected_stop_id})")

st.subheader("실제값 vs 예측값")
st.dataframe(compare_df.style.format({
    '실제 승차인원': '{:.0f}',
    '예측 승차인원': '{:.0f}'
}))

st.subheader("\U0001F4CA 예측 성능 지표")
st.markdown(f"- **MAE (평균 절대 오차)**: {mae:.2f}")
st.markdown(f"- **RMSE (평균 제곱근 오차)**: {rmse:.2f}")
st.markdown(f"- **R² Score (결정계수)**: {r2:.4f}")

# 그래프 출력
st.subheader("그래프 비교")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(compare_df['날짜'], compare_df['실제 승차인원'], label='실제값', marker='o')
ax.plot(compare_df['날짜'], compare_df['예측 승차인원'], label='예측값', marker='x')
ax.set_title(f"{selected_stop_name} (ID: {selected_stop_id}) - 실제값 vs 예측값")
ax.set_xlabel("날짜")
ax.set_ylabel("승차인원수")
ax.legend()
ax.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

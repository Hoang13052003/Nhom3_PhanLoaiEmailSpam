import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Đọc dữ liệu từ file CSV và thay đổi tên cột cho dễ hiểu
df = pd.read_csv("spam.csv", encoding='latin-1')[['Category', 'Message']]
df.columns = ['label', 'text']

# Chuyển đổi nhãn (label) từ 'ham'/'spam' thành 0/1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Tạo CountVectorizer (dùng stop-words tiếng Việt nếu có danh sách stop-words tiếng Việt)
vectorizer = CountVectorizer(stop_words=None)  # Thay 'None' nếu bạn muốn bỏ qua stop-words tiếng Anh

# Áp dụng vectorizer vào cột 'text' và chuyển dữ liệu thành ma trận số
X = vectorizer.fit_transform(df['text'])

# Biến mục tiêu (y) là nhãn 'label'
y = df['label']

# Khởi tạo mô hình Naïve Bayes
model = MultinomialNB()

# Huấn luyện mô hình với dữ liệu
model.fit(X, y)

# Lưu mô hình và vectorizer để sử dụng sau
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

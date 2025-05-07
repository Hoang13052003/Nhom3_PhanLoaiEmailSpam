from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import traceback # Để debug lỗi chi tiết hơn

app = Flask(__name__)
# Cấu hình CORS để cho phép frontend từ cổng 3000 truy cập (cổng mặc định của create-react-app/Vite dev server)
# Trong môi trường production, bạn cần cấu hình domain cụ thể thay vì "*"
CORS(app)

# --- Load mô hình và vectorizer ---
# Đảm bảo rằng các file này tồn tại và có thể truy cập được
try:
    model = joblib.load("spam_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    print("Mô hình và vectorizer đã được tải thành công.")
except FileNotFoundError:
    print("LỖI: Không tìm thấy file 'spam_model.pkl' hoặc 'vectorizer.pkl'. Hãy đảm bảo chúng nằm cùng thư mục với app.py.")
    model = None
    vectorizer = None
except Exception as e:
    print(f"LỖI khi tải mô hình/vectorizer: {e}")
    model = None
    vectorizer = None


@app.route('/classify', methods=['POST'])
def classify():
    if model is None or vectorizer is None:
        return jsonify({"error": "Mô hình hoặc vectorizer chưa được tải. Vui lòng kiểm tra log server."}), 500

    # --- Kiểm tra và lấy input ---
    data = request.get_json()
    if not data or 'email' not in data:
        return jsonify({"error": "Dữ liệu đầu vào không hợp lệ. Vui lòng gửi JSON object với trường 'email'."}), 400

    email = data['email']
    if not isinstance(email, str) or not email.strip():
         return jsonify({"error": "Nội dung email không được rỗng."}), 400

    try:
        # --- Xử lý phân loại ---
        # vectorizer.transform tự động áp dụng các bước preprocessing đã được dùng khi train
        vec = vectorizer.transform([email])

        # Kiểm tra nếu vector kết quả rỗng (có thể do email chỉ chứa stop words hoặc ký tự bị loại bỏ)
        if vec.nnz == 0: # nnz counts non-zero elements
             return jsonify({
                "result": "Không rõ", # Hoặc một label phù hợp khác
                "spam_prob": 50.0, # Xác suất 50/50
                "ham_prob": 50.0,
                "top_words": [], # Không có từ nào có nghĩa
                "message": "Email không chứa các từ khóa đã học."
             }), 200


        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0] # prob[0] là xác suất Ham, prob[1] là xác suất Spam

        # --- Trích xuất và tính trọng số từ khóa ---
        feature_names = vectorizer.get_feature_names_out()
        email_vector_dense = vec.toarray()[0] # Chuyển sparse matrix thành dense array

        # Chỉ lấy các chỉ mục của các từ xuất hiện (có giá trị > 0 trong vector)
        word_indices_in_email = np.where(email_vector_dense > 0)[0]

        word_weights = []

        for idx in word_indices_in_email:
            word = feature_names[idx]
            # Lấy log xác suất từ mô hình Naive Bayes
            # model.feature_log_prob_ có shape (n_classes, n_features)
            # model.feature_log_prob_[0] là log-prob cho lớp Ham
            # model.feature_log_prob_[1] là log-prob cho lớp Spam
            try:
                spam_log_prob = model.feature_log_prob_[1][idx]
                ham_log_prob = model.feature_log_prob_[0][idx]
                 # Lưu ý: có thể cần xử lý smoothing nếu bạn dùng Additive Smoothing (Laplace).
                 # feature_log_prob_ đã bao gồm smoothing nếu mô hình được khởi tạo với alpha > 0.

                difference = spam_log_prob - ham_log_prob

                word_weights.append({
                    "word": word,
                    "spam_weight": float(spam_log_prob), # Gửi float thay vì round ở đây, frontend sẽ format
                    "ham_weight": float(ham_log_prob),
                    "difference": float(difference)
                })
            except IndexError:
                 # Trường hợp hiếm: từ có trong vectorizer nhưng không có log_prob trong model (lạ)
                 print(f"Cảnh báo: Không tìm thấy log_prob cho chỉ mục {idx} (từ: {word})")
                 continue # Bỏ qua từ này

        # Sắp xếp theo độ chênh lệch log-prob tuyệt đối giảm dần
        # Lấy top N từ có ảnh hưởng nhất (cả spam và ham)
        top_words = sorted(word_weights, key=lambda x: abs(x["difference"]), reverse=True)[:15] # Lấy nhiều hơn 5 cho frontend chọn

        # --- Trả về kết quả ---
        return jsonify({
            "result": "Spam" if pred == 1 else "Không Spam",
            "spam_prob": round(prob[1] * 100, 2),
            "ham_prob": round(prob[0] * 100, 2),
            "top_words": top_words # Danh sách các từ khóa có trọng số
        })

    except Exception as e:
        # Ghi log lỗi chi tiết trên server
        traceback.print_exc()
        return jsonify({"error": f"Đã xảy ra lỗi nội bộ: {e}"}), 500

if __name__ == '__main__':
    # Khi debug=True, Flask server sẽ tự động reload khi có thay đổi
    # Trong production, bạn nên sử dụng một WSGI server như Gunicorn hoặc uWSGI
    app.run(debug=True, port=5000)
import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# 1. Tải lại mô hình đã lưu
model = tf.keras.models.load_model("MS_MobileNetV2.keras")

# ⚠️ QUAN TRỌNG: Danh sách nhãn phải ĐÚNG THỨ TỰ với lúc train (thứ tự chữ cái của tên thư mục)
# Bạn có thể in ra class_names ở file train_model.py để điền chính xác vào đây:
class_names = [
    "bè",
    "thuyền"
]  # Ví dụ (thay thế bằng danh sách thực tế của bạn)


# 2. Viết hàm xử lý ảnh và dự đoán
def classify_image(img):
  # Resize ảnh về chuẩn 224x224 mà MobileNetV2 yêu cầu
  img = img.resize((224, 224))

  # Chuyển thành mảng numpy và thêm chiều batch
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = x / 255.0  # Chuẩn hóa pixel về khoảng [0, 1]

  # Dự đoán
  predictions = model.predict(x)
  score = tf.nn.softmax(predictions[0])

  # Trả về kết quả dạng dictionary để Gradio vẽ biểu đồ xác suất
  return {
      class_names[i]: float(score[i]) for i in range(len(class_names))
  }


# 3. Thiết kế giao diện kéo-thả với Gradio
demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(
        type="pil", label="Kéo thả hoặc tải ảnh bất kỳ vào đây"
    ),
    outputs=gr.Label(
        num_top_classes=3, label="Kết quả dự đoán độ tin cậy"
    ),
    title="Phân loại bè với thuyền bằng MobileNetV2",
    description="Tải lên một bức ảnh để mô hình phân loại xem thuộc nhãn nào.",
)

if __name__ == "__main__":
  # Khởi chạy giao diện web cục bộ
  demo.launch(share=True)
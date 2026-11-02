import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# 1. Cấu hình thông số
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# Đường dẫn dữ liệu (tương đối theo thư mục hiện tại)
train_dir = r"D:\my repo\training mobilenetv2\Training-MobileNetV2-to-predict\dataset\train"
val_dir = r"D:\my repo\training mobilenetv2\Training-MobileNetV2-to-predict\dataset\valid"

# 2. Tải dữ liệu từ thư mục
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    val_dir, shuffle=False, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE
)

# Lấy danh sách tên các nhãn (classes) tự động từ tên thư mục
class_names = train_dataset.class_names
print("Các nhãn phát hiện được:", class_names)

# Tối ưu hóa tốc độ nạp dữ liệu
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

# 3. Xây dựng mô hình với MobileNetV2 (Transfer Learning)
base_model = MobileNetV2(
    input_shape=(224, 224, 3), include_top=False, weights="imagenet"
)
base_model.trainable = (
    False  # Đóng băng phần trích xuất đặc trưng của ImageNet
)

model = models.Sequential(
    [
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(
            len(class_names), activation="softmax"
        ),  # Số đầu ra = số lượng nhãn
    ]
)

# 4. Biên dịch mô hình
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",  # Dùng sparse vì nhãn ở dạng số nguyên tự nhiên từ image_dataset_from_directory
    metrics=["accuracy"],
)

# 5. Huấn luyện mô hình
print("Đang bắt đầu huấn luyện mô hình...")
history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS)

# 6. Lưu mô hình thành 1 file duy nhất
model.save("my_model.keras")
print("Đã huấn luyện xong và lưu mô hình tại: my_model.keras")
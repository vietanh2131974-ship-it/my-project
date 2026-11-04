import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import h5py
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import sys
import os


# ==================== ĐỊNH NGHĨA MÔ HÌNH ====================
class CNNModel(nn.Module):
    def __init__(self, num_classes=3):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ==================== CẤU HÌNH ====================
CONFIDENCE_THRESHOLD = 0.5  # Ngưỡng tin cậy


# ==================== KIỂM TRA FILE ====================
def check_files():
    if not os.path.exists("vehicle_model.h5"):
        messagebox.showerror("Lỗi", "Không tìm thấy file 'vehicle_model.h5'!\nHãy đảm bảo bạn đã train và lưu model.")
        return False

    if not os.path.exists("labels.txt"):
        # Tạo file labels.txt mặc định nếu chưa có
        default_labels = ["xe_may", "xe_oto", "xe_tai"]
        with open("labels.txt", "w", encoding="utf-8") as f:
            for label in default_labels:
                f.write(label + "\n")
        print("✅ Đã tạo file labels.txt mặc định")

    return True


# ==================== TẢI NHÃN ====================
def load_labels():
    with open("labels.txt", "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f]
    return class_names


# ==================== TẢI MÔ HÌNH ====================
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📟 Đang dùng thiết bị: {device}")

    class_names = load_labels()
    model = CNNModel(num_classes=len(class_names))

    print("📦 Đang load trọng số từ vehicle_model.h5...")
    try:
        state_dict = {}
        with h5py.File("vehicle_model.h5", "r") as hdf:
            for key in hdf.keys():
                state_dict[key] = torch.tensor(hdf[key][()])

        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        print("✅ Đã load mô hình thành công!")
        return model, device, class_names
    except Exception as e:
        messagebox.showerror("Lỗi", f"Lỗi khi load model: {e}")
        return None, None, None


# ==================== TIỀN XỬ LÝ ====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# ==================== HÀM DỰ ĐOÁN ====================
def predict_image(model, device, image_path, class_names):
    """Dự đoán từ file ảnh"""
    try:
        # Đọc ảnh
        image = Image.open(image_path).convert('RGB')

        # Tiền xử lý
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Dự đoán
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        return class_names[predicted.item()], confidence.item(), probabilities[0]
    except Exception as e:
        print(f"Lỗi khi dự đoán: {e}")
        return None, 0, None


# ==================== HIỂN THỊ KẾT QUẢ ====================
def show_result(image_path, prediction, confidence, probabilities, class_names):
    """Hiển thị ảnh và kết quả dự đoán trong cửa sổ mới"""
    result_window = tk.Toplevel(root)
    result_window.title("Kết quả nhận diện")
    result_window.geometry("500x600")
    result_window.resizable(False, False)

    # Hiển thị ảnh
    img = Image.open(image_path)
    img.thumbnail((400, 400))
    photo = ImageTk.PhotoImage(img)

    img_label = tk.Label(result_window, image=photo)
    img_label.image = photo
    img_label.pack(pady=10)

    # Hiển thị kết quả
    if confidence < CONFIDENCE_THRESHOLD:
        result_text = f"🔍 Không đủ tin cậy\n(Độ tin cậy: {confidence * 100:.1f}%)"
        result_color = "orange"
    else:
        result_text = f"✅ Dự đoán: {prediction.upper()}\n📊 Độ tin cậy: {confidence * 100:.1f}%"
        result_color = "green"

    result_label = tk.Label(result_window, text=result_text,
                            font=("Helvetica", 16, "bold"),
                            fg=result_color)
    result_label.pack(pady=10)

    # Hiển thị chi tiết xác suất từng lớp
    detail_frame = tk.Frame(result_window)
    detail_frame.pack(pady=10, padx=20, fill="both")

    tk.Label(detail_frame, text="Chi tiết xác suất:",
             font=("Helvetica", 12, "bold")).pack()

    for i, class_name in enumerate(class_names):
        prob = probabilities[i].item() * 100
        prob_text = f"{class_name}: {prob:.1f}%"
        prob_color = "blue" if i == probabilities.argmax().item() else "black"

        tk.Label(detail_frame, text=prob_text,
                 font=("Helvetica", 11),
                 fg=prob_color).pack()

    # Nút đóng
    tk.Button(result_window, text="Đóng", command=result_window.destroy,
              font=("Helvetica", 12), bg="red", fg="white").pack(pady=20)


# ==================== CHỌN ẢNH TỪ MÁY TÍNH ====================
def select_image():
    """Mở hộp thoại chọn file ảnh"""
    file_path = filedialog.askopenfilename(
        title="Chọn ảnh cần nhận diện",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
            ("All files", "*.*")
        ]
    )

    if not file_path:
        return

    # Cập nhật đường dẫn trên giao diện
    path_label.config(text=f"📁 {os.path.basename(file_path)}")

    # Dự đoán
    status_label.config(text="🔄 Đang xử lý...", fg="orange")
    root.update()

    prediction, confidence, probabilities = predict_image(model, device, file_path, class_names)

    if prediction:
        status_label.config(text="✅ Nhận diện hoàn tất!", fg="green")
        show_result(file_path, prediction, confidence, probabilities, class_names)
    else:
        status_label.config(text="❌ Lỗi khi nhận diện!", fg="red")
        messagebox.showerror("Lỗi", "Không thể nhận diện ảnh này!")


# ==================== CHỌN NHIỀU ẢNH ====================
def select_multiple_images():
    """Chọn nhiều ảnh cùng lúc"""
    file_paths = filedialog.askopenfilenames(
        title="Chọn nhiều ảnh cần nhận diện",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
            ("All files", "*.*")
        ]
    )

    if not file_paths:
        return

    path_label.config(text=f"📁 Đã chọn {len(file_paths)} ảnh")
    status_label.config(text="🔄 Đang xử lý...", fg="orange")
    root.update()

    # Tạo cửa sổ kết quả tổng hợp
    result_window = tk.Toplevel(root)
    result_window.title("Kết quả nhận diện nhiều ảnh")
    result_window.geometry("600x500")

    # Text widget để hiển thị kết quả
    text_widget = tk.Text(result_window, wrap=tk.WORD, font=("Courier", 10))
    text_widget.pack(fill="both", expand=True, padx=10, pady=10)

    scrollbar = tk.Scrollbar(text_widget)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    text_widget.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=text_widget.yview)

    # Dự đoán từng ảnh
    results_text = "=" * 50 + "\n"
    results_text += f"KẾT QUẢ NHẬN DIỆN {len(file_paths)} ẢNH\n"
    results_text += "=" * 50 + "\n\n"

    for i, file_path in enumerate(file_paths, 1):
        prediction, confidence, probabilities = predict_image(model, device, file_path, class_names)

        results_text += f"{i}. {os.path.basename(file_path)}\n"
        if prediction and confidence >= CONFIDENCE_THRESHOLD:
            results_text += f"   ✅ {prediction.upper()} (độ tin cậy: {confidence * 100:.1f}%)\n"
        elif prediction:
            results_text += f"   ⚠️ {prediction.upper()} (độ tin cậy thấp: {confidence * 100:.1f}%)\n"
        else:
            results_text += f"   ❌ Không nhận diện được\n"
        results_text += "\n"

        # Cập nhật giao diện
        status_label.config(text=f"🔄 Đang xử lý ảnh {i}/{len(file_paths)}...", fg="orange")
        root.update()

    results_text += "=" * 50
    text_widget.insert(tk.END, results_text)
    text_widget.config(state=tk.DISABLED)

    status_label.config(text="✅ Nhận diện hoàn tất!", fg="green")

    # Nút lưu kết quả
    def save_results():
        save_path = filedialog.asksaveasfilename(
            title="Lưu kết quả",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")]
        )
        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(results_text)
            messagebox.showinfo("Thông báo", f"Đã lưu kết quả tại:\n{save_path}")

    tk.Button(result_window, text="💾 Lưu kết quả", command=save_results,
              font=("Helvetica", 10), bg="blue", fg="white").pack(pady=5)
    tk.Button(result_window, text="Đóng", command=result_window.destroy,
              font=("Helvetica", 10), bg="red", fg="white").pack(pady=5)


# ==================== GIAO DIỆN CHÍNH ====================
if __name__ == "__main__":
    # Kiểm tra file
    if not check_files():
        sys.exit(1)

    # Tải model
    model, device, class_names = load_model()
    if model is None:
        sys.exit(1)

    # Tạo giao diện
    root = tk.Tk()
    root.title("Nhận diện logo xe - Chọn ảnh từ máy tính")
    root.geometry("500x400")
    root.resizable(False, False)

    # Tiêu đề
    title_label = tk.Label(root, text="🚗 NHẬN DIỆN LOGO XE 🚗",
                           font=("Helvetica", 18, "bold"),
                           fg="blue")
    title_label.pack(pady=20)

    # Frame chứa nút bấm
    button_frame = tk.Frame(root)
    button_frame.pack(pady=20)

    # Nút chọn 1 ảnh
    btn_single = tk.Button(button_frame, text="📷 CHỌN 1 ẢNH",
                           command=select_image,
                           font=("Helvetica", 14, "bold"),
                           bg="green", fg="white",
                           width=20, height=2)
    btn_single.pack(pady=10)

    # Nút chọn nhiều ảnh
    btn_multiple = tk.Button(button_frame, text="📁 CHỌN NHIỀU ẢNH",
                             command=select_multiple_images,
                             font=("Helvetica", 14, "bold"),
                             bg="orange", fg="white",
                             width=20, height=2)
    btn_multiple.pack(pady=10)

    # Hiển thị đường dẫn ảnh đã chọn
    path_label = tk.Label(root, text="Chưa chọn ảnh nào",
                          font=("Helvetica", 10),
                          fg="gray", wraplength=450)
    path_label.pack(pady=10)

    # Thông tin model
    info_frame = tk.Frame(root)
    info_frame.pack(pady=20)

    tk.Label(info_frame, text=f"📦 Model: CNN",
             font=("Helvetica", 9)).pack()
    tk.Label(info_frame, text=f"🏷️ Các nhãn: {', '.join(class_names)}",
             font=("Helvetica", 9)).pack()
    tk.Label(info_frame, text=f"⚙️ Ngưỡng tin cậy: {CONFIDENCE_THRESHOLD * 100:.0f}%",
             font=("Helvetica", 9)).pack()

    # Trạng thái
    status_label = tk.Label(root, text="✅ Sẵn sàng",
                            font=("Helvetica", 10),
                            fg="green")
    status_label.pack(pady=10)

    # Nút thoát
    btn_exit = tk.Button(root, text="Thoát",
                         command=root.quit,
                         font=("Helvetica", 12),
                         bg="red", fg="white",
                         width=15)
    btn_exit.pack(pady=10)

    print("\n" + "=" * 50)
    print("🎯 ỨNG DỤNG ĐÃ SẴN SÀNG!")
    print("📌 Click 'CHỌN 1 ẢNH' để nhận diện một ảnh")
    print("📌 Click 'CHỌN NHIỀU ẢNH' để nhận diện nhiều ảnh cùng lúc")
    print("=" * 50 + "\n")

    root.mainloop()
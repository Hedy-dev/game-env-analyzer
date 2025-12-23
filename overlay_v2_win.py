import tkinter as tk
import threading
import time
import sys
import argparse
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image as PILImage
from datetime import datetime
from collections import OrderedDict
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from mss import mss
from ultralytics import YOLO
import torchvision.models as models
FPS = 0
LAST_FRAME_TIME = None
def get_pretrained_model(num_classes):
    model = models.resnet18(pretrained=True)
    # Глобально размораживаем, так как датасет специфичный
    for param in model.parameters():
        param.requires_grad = True

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model
# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1, downsample=None):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
#         self.batch1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=1, bias=False)
#         self.batch2 = nn.BatchNorm2d(out_channels)
#         self.down_sample = nn.Identity()
#         if in_channels != out_channels or stride != 1:
#             self.down_sample = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=1, padding=0, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )

#     def forward(self, x):
#         identity = x
#         if self.down_sample is not None:
#             identity = self.down_sample(x)
#         out = self.conv1(x)
#         out = self.batch1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.batch2(out)
#         out += identity
#         out = self.relu(out)
#         return out

# class CustomLightNet(nn.Module):
#     def __init__(self, num_classes=5):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.relu = nn.LeakyReLU(inplace=True)
#         self.basic1 = ResidualBlock(16, 16)
#         self.basic2 = ResidualBlock(16, 16) 

#         self.basic3 = ResidualBlock(16, 32, stride=2)
#         self.basic4 = ResidualBlock(32, 32, stride=1) 

#         self.basic5 = ResidualBlock(32, 64, stride=2)
#         self.basic6 = ResidualBlock(64, 64, stride=1) 

#         self.basic7 = ResidualBlock(64, 128, stride=2)
#         self.basic8 = ResidualBlock(128, 128, stride=1) 

#         self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
#         # Dropout
#         self.dropout = nn.Dropout(p=0.5)
#         self.fc = nn.Linear(128, num_classes)
#         self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.relu((self.bn1((self.conv1((x))))))
        out = self.basic2(self.basic1(out))
        out = self.basic4(self.basic3(out))
        out = self.basic6(self.basic5(out))
        out = self.basic8(self.basic7(out))
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

def load_yolo_model(model_path):
    try:
        model = YOLO(model_path)
        print(f"YOLO модель загружена из {model_path}")
        return model
    except Exception as e:
        print(f"Ошибка при загрузке YOLO модели: {e}")
        return None

# def load_pytorch_model(model_path, num_classes=14):
#     try:
#         # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         # print(f"Используется устройство: {device}")
#         # checkpoint = torch.load(model_path, map_location=device)
#         # print(f"Тип загруженных данных: {type(checkpoint)}")
#         # if num_classes is None:
#         #     if isinstance(checkpoint, OrderedDict):
#         #         fc_weight_key = None
#         #         fc_bias_key = None
#         #         for key in checkpoint.keys():
#         #             if 'fc.weight' in key:
#         #                 fc_weight_key = key
#         #             elif 'fc.bias' in key:
#         #                 fc_bias_key = key
#         #         if fc_weight_key and fc_weight_key in checkpoint:
#         #             weight_shape = checkpoint[fc_weight_key].shape
#         #             if len(weight_shape) == 2:
#         #                 num_classes = weight_shape[0]
#         #                 print(f"Определено количество классов: {num_classes} (из весов fc.weight)")
#         #             else:
#         #                 num_classes = 5
#         #         else:
#         #             num_classes = 5
#         #     elif isinstance(checkpoint, dict):
#         #         if 'state_dict' in checkpoint:
#         #             state_dict = checkpoint['state_dict']
#         #             for key, value in state_dict.items():
#         #                 if 'fc.weight' in key:
#         #                     weight_shape = value.shape
#         #                     if len(weight_shape) == 2:
#         #                         num_classes = weight_shape[0]
#         #                         print(f"Определено количество классов: {num_classes} (из checkpoint['state_dict'])")
#         #                         break
#         #         else:
#         #             num_classes = 5
#         #     else:
#         #         num_classes = 5
#         # else:
#         #     print(f"Используется указанное количество классов: {num_classes}")
        
#         # model = CustomLightNet(num_classes=num_classes)
        
#         # if isinstance(checkpoint, OrderedDict):
#         #     print(f"Загружается state_dict ({len(checkpoint)} параметров)")
#         #     model.load_state_dict(checkpoint)
#         # elif isinstance(checkpoint, dict):
#         #     if 'state_dict' in checkpoint:
#         #         print(f"Загружается state_dict из checkpoint")
#         #         model.load_state_dict(checkpoint['state_dict'])
#         #     elif 'model' in checkpoint:
#         #         print(f"Загружается модель из checkpoint с ключом 'model'")
#         #         model.load_state_dict(checkpoint['model'])
#         #     else:
#         #         try:
#         #             print(f"Попытка загрузить как state_dict напрямую")
#         #             model.load_state_dict(checkpoint)
#         #         except Exception as e:
#         #             print(f"Не удалось загрузить как state_dict: {e}")
#         #             return None, None, num_classes
#         # else:
#         #     print(f"Загружается полная модель")
#         #     model = checkpoint
        
#         # model = model.to(device)
        
#         # model.eval()
#         model =  get_pretrained_model(num_classes=num_classes) 
#         model.load_state_dict(torch.load('best_action_model_multi_class_last.pth'))
#         model = model.to(device)
#         model.eval()

#         print("модель успешно загружена из файла.")
#         print(f"PyTorch модель успешно загружена из {model_path}")
#         print(f"Архитектура модели: {type(model).__name__}")
#         print(f"Количество классов: {num_classes}")
        
#         total_params = sum(p.numel() for p in model.parameters())
#         print(f"Общее количество параметров: {total_params:,}")
        
#         return model, device, num_classes
#     except Exception as e:
#         print(f"Ошибка при загрузке PyTorch модели: {e}")
#         import traceback
#         traceback.print_exc()
#         print("PyTorch модель не будет использоваться")
#         return None, None, num_classes

def load_pytorch_model(model_path, num_classes=14):
    try:
        # FIX 1: Define device (it was commented out in your code)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Используется устройство: {device}")

        # FIX 2: Handle None explicitly. 
        # If args.num_classes is None, it overrides the default=14 in the signature, so we must check here.
        if num_classes is None:
            num_classes = 14
            print(f"Количество классов не указано, используется по умолчанию: {num_classes}")

        # Initialize model structure
        model = get_pretrained_model(num_classes=num_classes)
        
        # FIX 3: Use the 'model_path' argument instead of the hardcoded filename
        print(f"Загрузка весов из: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)

        # Handle different saving formats (state_dict vs full checkpoint)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model = model.to(device)
        model.eval()

        print("Модель успешно загружена из файла.")
        print(f"Архитектура модели: {type(model).__name__}")
        print(f"Количество классов: {num_classes}")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Общее количество параметров: {total_params:,}")
        
        return model, device, num_classes

    except Exception as e:
        print(f"Ошибка при загрузке PyTorch модели: {e}")
        import traceback
        traceback.print_exc()
        print("PyTorch модель не будет использоваться")
        return None, None, num_classes

def create_parser():
    parser = argparse.ArgumentParser(description='Двойной детектор Overlay')
    parser.add_argument('yolo_model_path', type=str, 
                       help='Путь к файлу модели YOLO с расширением .pt')
    parser.add_argument('pytorch_model_path', type=str, nargs='?', default=None,
                       help='Путь к файлу модели PyTorch с расширением .pth (опционально)')
    parser.add_argument('--num-classes', type=int, default=None,
                       help='Количество классов для PyTorch модели (определяется автоматически если не указано)')
    parser.add_argument('--pytorch-classes-file', type=str, default=None,
                       help='Файл с именами классов PyTorch модели (по одной строке на класс)')
    parser.add_argument('--show-top-n', type=int, default=3,
                       help='Количество классов для отображения (по умолчанию: 3)')
    return parser

def calculate_fps():
    global LAST_FRAME_TIME, FPS
    current_time = time.time()
    if LAST_FRAME_TIME is not None:
        time_diff = current_time - LAST_FRAME_TIME
        if time_diff > 0:
            FPS = 1.0 / time_diff
    LAST_FRAME_TIME = current_time
    return FPS

# def extract_and_prepare_bbox(frame, bbox, target_size=(224, 224)):
#     try:
#         x1, y1, x2, y2 = bbox.astype(int)
#         h, w = frame.shape[:2]
#         x1 = max(0, x1)
#         y1 = max(0, y1)
#         x2 = min(w, x2)
#         y2 = min(h, y2)
#         if x2 <= x1 or y2 <= y1:
#             return None
#         region = frame[y1:y2, x1:x2]
#         region_rgb = region[..., ::-1]
#         pil_image = PILImage.fromarray(region_rgb)
#         # transform = transforms.Compose([
#         #     transforms.Resize(target_size),
#         #     transforms.ToTensor(),
#         #     # transforms.Normalize(mean=[0.485, 0.456, 0.406], 
#         #     #                    std=[0.229, 0.224, 0.225])
#         # ])
#         transform = A.Compose([
#         A.Resize(224, 224),
#         A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ToTensorV2()
#         ])

#         tensor_image = transform(pil_image)
#         return tensor_image.unsqueeze(0)
#     except Exception as e:
#         print(f"Ошибка при подготовке bbox: {e}")
#         return None
def extract_and_prepare_bbox(frame, bbox, target_size=(224, 224)):
    try:
        x1, y1, x2, y2 = bbox.astype(int)
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
            
        region = frame[y1:y2, x1:x2]
        region_rgb = region[..., ::-1]  # Конвертация BGR -> RGB (numpy array)
        
        # PIL Image здесь не нужен, Albumentations работает с numpy array напрямую
        
        transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


        augmented = transform(image=region_rgb)
        tensor_image = augmented['image']
        
        return tensor_image.unsqueeze(0)
    except Exception as e:
        print(f"Ошибка при подготовке bbox: {e}")
        return None

def load_class_names(filepath, expected_num_classes=None):
    if filepath and os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                class_names = [line.strip() for line in f if line.strip()]
            if expected_num_classes and len(class_names) != expected_num_classes:
                print(f"Внимание: В файле {len(class_names)} классов, ожидается {expected_num_classes}")
                if len(class_names) > expected_num_classes:
                    class_names = class_names[:expected_num_classes]
                    print(f"Список классов обрезан до {expected_num_classes}")
                else:
                    for i in range(len(class_names), expected_num_classes):
                        class_names.append(f"Class_{i}")
                    print(f"Список классов дополнен до {expected_num_classes}")
            print(f"Загружены имена классов из {filepath}: {len(class_names)} классов")
            return class_names
        except Exception as e:
            print(f"Ошибка при загрузке имен классов: {e}")
    return None

def pytorch_model_prediction(model, device, tensor_image, class_names=None, top_n=3):
    """
    Выполняет предсказание с помощью PyTorch модели
    Возвращает все предсказания с уверенностью
    """
    try:
        with torch.no_grad():
            tensor_image = tensor_image.to(device)
            output = model(tensor_image)
            if isinstance(output, (list, tuple)):
                output = output[0]
            probabilities = torch.nn.functional.softmax(output, dim=1)
            num_classes = probabilities.size(1)
            all_probs, all_indices = torch.topk(probabilities, k=num_classes)
            results = []
            for i in range(num_classes):
                class_idx = all_indices[0][i].item()
                confidence = all_probs[0][i].item()
                if class_names and class_idx < len(class_names):
                    class_name = class_names[class_idx]
                else:
                    class_name = f"Class_{class_idx}"
                results.append({
                    'class_idx': class_idx,
                    'class_name': class_name,
                    'confidence': confidence,
                    'rank': i + 1
                })
            return results
    except Exception as e:
        print(f"Ошибка при предсказании PyTorch модели: {e}")
        return None

def model_predictions(yolo_model, pytorch_model_info, frame: np.ndarray, pytorch_num_classes=None, pytorch_class_names=None, top_n=3):
    yolo_detections = []
    pytorch_predictions = []
    if yolo_model is None:
        h, w, _ = frame.shape
        return yolo_detections, pytorch_predictions
    try:
        frame_rgb = frame[..., ::-1]  # BGR -> RGB
        results = yolo_model(frame_rgb, verbose=False)
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                max_detections = min(3, len(result.boxes))
                for i in range(max_detections):
                    box = result.boxes[i]
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy()
                    yolo_detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': bbox,
                        'id': i
                    })
                    if pytorch_model_info is not None:
                        pytorch_model, device, num_classes = pytorch_model_info
                        tensor_image = extract_and_prepare_bbox(frame, bbox, target_size=(224,224))
                        if tensor_image is not None:
                            pytorch_preds = pytorch_model_prediction(
                                pytorch_model, device, tensor_image, 
                                pytorch_class_names, top_n
                            )
                            if pytorch_preds:
                                pytorch_predictions.append({
                                    'yolo_class': class_name,
                                    'yolo_confidence': confidence,
                                    'all_predictions': pytorch_preds
                                })
        return yolo_detections, pytorch_predictions
    except Exception as e:
        print(f"Ошибка при предсказаниях: {e}")
        return [], []

class Overlay(tk.Tk):
    def __init__(self, yolo_model, pytorch_model_info, pytorch_class_names=None, show_top_n=3):
        super().__init__()
        self.yolo_model = yolo_model
        self.pytorch_model_info = pytorch_model_info
        self.pytorch_class_names = pytorch_class_names
        self.show_top_n = show_top_n
        self.has_pytorch_model = pytorch_model_info is not None
        if self.has_pytorch_model:
            self.pytorch_model, self.pytorch_device, self.pytorch_num_classes = pytorch_model_info
        else:
            self.pytorch_num_classes = 0

        self.title("Двойной детектор")
        self.geometry("700x300+50+50")
        self.attributes("-topmost", True)
        self.attributes("-alpha", 0.85)
        self.overrideredirect(True)
        main_frame = tk.Frame(self, bg="black")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        self.fps_label = tk.Label(
            main_frame,
            text="FPS: 0.0",
            fg="yellow",
            bg="black",
            font=("Arial", 12, "bold")
        )
        self.fps_label.pack(anchor=tk.W, pady=(0, 5))
        self.yolo_label = tk.Label(
            main_frame,
            text="YOLO: ожидание...",
            fg="cyan",
            bg="black",
            font=("Arial", 11),
            wraplength=660,
            justify=tk.LEFT
        )
        self.yolo_label.pack(anchor=tk.W, pady=(0, 5))
        if self.has_pytorch_model:
            self.pytorch_main_label = tk.Label(
                main_frame,
                text="PyTorch: ожидание...",
                fg="lightgreen",
                bg="black",
                font=("Arial", 11, "bold"),
                wraplength=660,
                justify=tk.LEFT
            )
            self.pytorch_main_label.pack(anchor=tk.W, pady=(0, 5))
            self.pytorch_other_label = tk.Label(
                main_frame,
                text="Другие классы: ожидание...",
                fg="lightblue",
                bg="black",
                font=("Arial", 10),
                wraplength=660,
                justify=tk.LEFT
            )
            self.pytorch_other_label.pack(anchor=tk.W, pady=(0, 5))
        status_text = f"Модели: YOLO {'✓' if yolo_model else '✗'}"
        if self.has_pytorch_model:
            status_text += f" | PyTorch ✓ (классов: {self.pytorch_num_classes}, показываем: {show_top_n})"
            if pytorch_class_names:
                status_text += f" | Имена классов: загружены"
        self.status_label = tk.Label(
            main_frame,
            text=status_text,
            fg="white",
            bg="black",
            font=("Arial", 10)
        )
        self.status_label.pack(anchor=tk.W)

        self.bind("<ButtonPress-1>", self.start_move)
        self.bind("<B1-Motion>", self.do_move)
        self.bind("<Double-Button-1>", lambda e: self.on_close())

        self.running = True
        threading.Thread(target=self.capture_loop, daemon=True).start()

    def start_move(self, e):
        self._x = e.x
        self._y = e.y

    def do_move(self, e):
        x = self.winfo_x() + e.x - self._x
        y = self.winfo_y() + e.y - self._y
        self.geometry(f"+{x}+{y}")

    def format_predictions_text(self, yolo_detections, pytorch_predictions):
        """Форматирует текст предсказаний для отображения"""
        yolo_texts = []
        pytorch_main_text = ""
        pytorch_other_text = ""
        for det in yolo_detections[:3]:
            yolo_texts.append(f"{det['class']}:{det['confidence']:.2f}")
        yolo_display = "YOLO: " + (", ".join(yolo_texts) if yolo_texts else "нет детекций")
        if self.has_pytorch_model and pytorch_predictions:
            pred_info = pytorch_predictions[0]
            all_preds = pred_info['all_predictions']
            if all_preds:
                main_pred = all_preds[0]
                pytorch_main_text = f"PyTorch: {pred_info['yolo_class']}→{main_pred['class_name']}:{main_pred['confidence']:.2f}"
                other_preds_text = []
                for i in range(1, min(self.show_top_n, len(all_preds))):
                    pred = all_preds[i]
                    other_preds_text.append(f"{pred['class_name']}:{pred['confidence']:.2f}")
                if other_preds_text:
                    pytorch_other_text = "Другие: " + ", ".join(other_preds_text)
                else:
                    pytorch_other_text = "Другие: нет"
        else:
            pytorch_main_text = "PyTorch: нет предсказаний" if self.has_pytorch_model else ""
            pytorch_other_text = ""
        return yolo_display, pytorch_main_text, pytorch_other_text

    def capture_loop(self):
        with mss() as sct:
            monitor = sct.monitors[1]

            while self.running:
                try:
                    fps = calculate_fps()
                    sct_img = sct.grab(monitor)
                    frame = np.array(sct_img)[:, :, :3]
                    yolo_detections, pytorch_predictions = model_predictions(
                        self.yolo_model, self.pytorch_model_info, frame, 
                        self.pytorch_num_classes, self.pytorch_class_names,
                        self.show_top_n
                    )
                    yolo_display, pytorch_main, pytorch_other = self.format_predictions_text(
                        yolo_detections, pytorch_predictions
                    )
                    self.after(0, self.update_display, fps, yolo_display, pytorch_main, pytorch_other)
                except Exception as e:
                    error_text = f"Ошибка: {str(e)[:40]}"
                    self.after(0, self.yolo_label.config, {"text": error_text})
                    time.sleep(1)

    def update_display(self, fps, yolo_display, pytorch_main, pytorch_other):
        """Обновляет отображение в основном потоке"""
        self.fps_label.config(text=f"FPS: {fps:.1f}")
        self.yolo_label.config(text=yolo_display)
        if self.has_pytorch_model:
            self.pytorch_main_label.config(text=pytorch_main)
            self.pytorch_other_label.config(text=pytorch_other)

    def on_close(self):
        self.running = False
        print("Программа завершена")
        self.destroy()


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    if not os.path.exists(args.yolo_model_path):
        print(f"Ошибка: Файл YOLO модели не найден: {args.yolo_model_path}")
        sys.exit(1)
    yolo_model = load_yolo_model(args.yolo_model_path)
    pytorch_model_info = None
    pytorch_num_classes = None
    if args.pytorch_model_path:
        if os.path.exists(args.pytorch_model_path):
            pytorch_model, device, pytorch_num_classes = load_pytorch_model(
                args.pytorch_model_path, 
                num_classes=args.num_classes
            )
            if pytorch_model is not None:
                pytorch_model_info = (pytorch_model, device, pytorch_num_classes)
        else:
            print(f"Предупреждение: Файл PyTorch модели не найден: {args.pytorch_model_path}")
    pytorch_class_names = None
    if pytorch_model_info and args.pytorch_classes_file:
        pytorch_class_names = load_class_names(args.pytorch_classes_file, pytorch_num_classes)
    print("=" * 70)
    print(f"YOLO модель: {'загружена' if yolo_model else 'НЕ загружена'}")
    print(f"PyTorch модель: {'загружена' if pytorch_model_info else 'не используется'}")
    if pytorch_model_info:
        print(f"Количество классов: {pytorch_num_classes}")
        print(f"Показывать топ классов: {args.show_top_n}")
        if pytorch_class_names:
            print(f"Загружено имен классов: {len(pytorch_class_names)}")
    print("=" * 70)
    print("Управление:")
    print("- Перетаскивание: левая кнопка мыши")
    print("- Закрытие: двойной клик по окну")
    print("=" * 70)
    app = Overlay(yolo_model, pytorch_model_info, pytorch_class_names, args.show_top_n)
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()

import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from collections import deque
import numpy as np

# 1. بنية الموديل المتطابقة مع التدريب
class auto_movement(nn.Module):
    def __init__(self):
        super(auto_movement, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 4)   
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)
        x = x.view(-1, 128 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

# 2. تحميل الموديل والأوزان
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = auto_movement().to(device)
model.load_state_dict(torch.load('hand_model_4classes_V2.pth', map_location=device))
model.eval()

# 3. مصفوفة الكلاسات طبقاً للترتيب الأبجدي: F, L, O, W
classes_map = [
    {"label": "F", "action": "FORWARD"},
    {"label": "L", "action": "LEFT"},
    {"label": "O", "action": "STOP"},
    {"label": "W", "action": "RIGHT"}
]

# 4. معالجة الصور
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

cap = cv2.VideoCapture(0)
prob_history = deque(maxlen=6)

print("Starting Camera... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # مربع التركيز (ROI)
    x1, y1, x2, y2 = 140, 90, 380, 330
    roi = frame[y1:y2, x1:x2]

    # تحويل لـ RGB وتجهيز الـ Tensor
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(roi_rgb)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        raw_probs = torch.softmax(output, dim=1)[0].cpu().numpy()
        prob_history.append(raw_probs)
        smoothed = np.mean(prob_history, axis=0)

        # رفع حساسية كلاس O (STOP) بنسبة 25% لسرعة الالتقاط
        # الترتيب: 0: F, 1: L, 2: O, 3: W
        weighted_probs = smoothed.copy()
        weighted_probs[2] *= 1.25

        idx = int(np.argmax(weighted_probs))
        conf = smoothed[idx] * 100

    # رسم مستطيل اليد
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, "Put Hand Here", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # عرض احتمالات الحركات الأربعة على الشاشة
    y_pos = 35
    for i, cls in enumerate(classes_map):
        p = smoothed[i] * 100
        color = (0, 255, 0) if i == idx else (200, 200, 200)
        cv2.putText(frame, f"{cls['label']} ({cls['action']}): {p:.1f}%", 
                    (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2 if i == idx else 1)
        y_pos += 25

    # عتبة اتخاذ القرار (40% لتفادي UNCERTAIN وتثبيت القراءة)
    action_text = classes_map[idx]['action'] if conf >= 40 else "UNCERTAIN"
    cv2.putText(frame, f">> {action_text} <<", (x1, y2 + 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Hand Control System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
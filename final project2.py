import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from collections import deque
import numpy as np
import math

# 1. بنية الموديل
class auto_movement(nn.Module):
    def __init__(self):
        super(auto_movement, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 4)   
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)
        x = x.view(-1, 32 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# 2. تحميل الموديل
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = auto_movement().to(device)
model.load_state_dict(torch.load('hand_model_4classes.pth', map_location=device))
model.eval()

classes_map = [
    {"label": "F", "action": "FORWARD"},
    {"label": "L", "action": "LEFT"},
    {"label": "O", "action": "STOP"},
    {"label": "W", "action": "RIGHT"}
]

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

cap = cv2.VideoCapture(0)
prob_history = deque(maxlen=6)

# إعدادات نافذة المحاكاة
car_x, car_y = 250.0, 250.0
car_angle = -90.0
speed = 7.0
turn_speed = 6.0

print("Starting Camera & Simulation... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    x1, y1, x2, y2 = 140, 90, 380, 330
    roi_color = frame[y1:y2, x1:x2].copy()
    
    # --- التنبؤ بالحركة ---
    roi_rgb = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(roi_rgb)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        raw_probs = torch.softmax(output, dim=1)[0].cpu().numpy()
        prob_history.append(raw_probs)
        smoothed = np.mean(prob_history, axis=0)

        # تضخيم حساسية W لتقرأ بشكل أسرع (Index 3)
        weighted_probs = smoothed.copy()
        weighted_probs[3] *= 1.45  # زيادة حساسية W
        weighted_probs[1] *= 0.90  # تقليل حساسية L قليلاً لمنع التداخل
        
        idx = int(np.argmax(weighted_probs))
        conf = smoothed[idx] * 100

    action_text = classes_map[idx]['action'] if conf >= 40 else "UNCERTAIN"

    # --- تحديث حركة السيارة ---
    if action_text == "FORWARD":
        car_x += speed * math.cos(math.radians(car_angle))
        car_y += speed * math.sin(math.radians(car_angle))
    elif action_text == "LEFT":
        car_angle -= turn_speed
    elif action_text == "RIGHT":
        car_angle += turn_speed

    car_x %= 500
    car_y %= 500

    sim_frame = np.zeros((500, 500, 3), dtype=np.uint8)
    for i in range(0, 500, 50):
        cv2.line(sim_frame, (i, 0), (i, 500), (40, 40, 40), 1)
        cv2.line(sim_frame, (0, i), (500, i), (40, 40, 40), 1)

    rad = math.radians(car_angle)
    pt1 = [car_x + 20 * math.cos(rad), car_y + 20 * math.sin(rad)] 
    pt2 = [car_x + 15 * math.cos(rad + 2.5), car_y + 15 * math.sin(rad + 2.5)] 
    pt3 = [car_x + 15 * math.cos(rad - 2.5), car_y + 15 * math.sin(rad - 2.5)] 
    
    cv2.fillPoly(sim_frame, [np.array([pt1, pt2, pt3], np.int32)], (0, 255, 255))
    cv2.putText(sim_frame, f"Action: {action_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # --- رسم النقاط عبر عزل لون البشرة (HSV) بدلاً من الأبيض والأسود ---
    hsv_roi = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv_roi, lower_skin, upper_skin)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) > 1000:  
            hull = cv2.convexHull(cnt)
            cv2.drawContours(frame[y1:y2, x1:x2], [cnt], -1, (0, 255, 0), 1)
            for point in hull:
                cv2.circle(frame[y1:y2, x1:x2], (point[0][0], point[0][1]), 4, (0, 0, 255), -1)

    # --- واجهة الكاميرا ---
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(frame, "Put Hand Here", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    y_pos = 35
    for i, cls in enumerate(classes_map):
        p = smoothed[i] * 100
        color = (0, 255, 0) if i == idx else (200, 200, 200)
        cv2.putText(frame, f"{cls['label']} ({cls['action']}): {p:.1f}%", 
                    (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2 if i == idx else 1)
        y_pos += 25

    cv2.putText(frame, f">> {action_text} <<", (x1, y2 + 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Hand Control System", frame)
    cv2.imshow("Robot Simulation", sim_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
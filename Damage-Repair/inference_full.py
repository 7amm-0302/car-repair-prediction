"""
차량 파손 분석 통합 파이프라인
문경 YOLO + 세영 U-Net/ResNet34
"""

import os
import json
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
import torch.nn.functional as F
import numpy as np

# ===================
# 설정
# ===================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
THRESHOLD = 0.6

# 문경 YOLO 경로
YOLO_MODEL_PATH = r"C:\Users\swu\Desktop\car-repair-prediction\Yolo\model\best.pt"

# 세영 모델 경로
SEGMENTATION_MODEL_PATH = r"C:\Users\swu\Desktop\car-repair-prediction\Damage-Repair\model\best_segmentation_model.pth"
SEGMENTATION_CLASSES_PATH = r"C:\Users\swu\Desktop\car-repair-prediction\Damage-Repair\model\segmentation_classes.json"
REPAIR_MODEL_PATH = r"C:\Users\swu\Desktop\car-repair-prediction\Damage-Repair\model\best_repair_model.pth"
REPAIR_CLASSES_PATH = r"C:\Users\swu\Desktop\car-repair-prediction\Damage-Repair\model\repair_classes.json"

# 부위 클래스 (문경 YOLO)
PART_CLASSES = [
    "Front bumper", "Head lights", "Bonnet", "Windshield", "Roof",
    "Trunk lid", "Rear lamp", "Rear bumper", "Front fender", "Side mirror",
    "Front door", "Rear door", "Rocker panel", "A pillar", "B pillar",
    "C pillar", "Rear fender", "Front Wheel", "Rear Wheel", "Rear windshield",
    "Undercarriage"
]

# ===================
# 세영 모델 정의 (학습 시 구조와 동일)
# ===================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=5):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024)
        
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        
        d4 = self.dec4(torch.cat([self.upconv4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.upconv3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upconv1(d2), e1], dim=1))
        
        return self.out_conv(d1)


class RepairMultiLabelClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        resnet = models.resnet34(weights=None)
        
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ===================
# 통합 파이프라인
# ===================

class FullPipeline:
    def __init__(self):
        self.device = DEVICE
        self.threshold = THRESHOLD
        
        # 전처리
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 문경 YOLO 로드
        print("Loading YOLO (문경)...")
        from ultralytics import YOLO
        self.yolo = YOLO(YOLO_MODEL_PATH)
        print("  [OK] YOLO loaded")
        
        # 세영 세그멘테이션 로드
        print("Loading U-Net (세영)...")
        with open(SEGMENTATION_CLASSES_PATH, 'r', encoding='utf-8') as f:
            seg_info = json.load(f)
        
        self.damage_types = seg_info["damage_types"]
        self.seg_classes = seg_info["num_classes"]
        self.idx_to_damage = {v: k for k, v in seg_info["class_mapping"].items()}
        self.idx_to_damage[0] = "Background"
        
        self.seg_model = UNet(in_channels=3, num_classes=self.seg_classes)
        self.seg_model.load_state_dict(torch.load(SEGMENTATION_MODEL_PATH, map_location=self.device))
        self.seg_model.to(self.device)
        self.seg_model.eval()
        print(f"  [OK] U-Net ({self.seg_classes} classes)")
        
        # 세영 수리 분류 로드
        print("Loading RepairClassifier (세영)...")
        with open(REPAIR_CLASSES_PATH, 'r', encoding='utf-8') as f:
            rep_info = json.load(f)
        
        self.repair_methods = rep_info["repair_methods"]
        
        self.rep_model = RepairMultiLabelClassifier(num_classes=len(self.repair_methods))
        self.rep_model.load_state_dict(torch.load(REPAIR_MODEL_PATH, map_location=self.device))
        self.rep_model.to(self.device)
        self.rep_model.eval()
        print(f"  [OK] RepairClassifier ({len(self.repair_methods)} classes)")
    
    def detect_parts(self, img_path):
        """문경 YOLO로 부위 검출"""
        results = self.yolo(img_path, conf=0.15)
        
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                bbox = box.xyxy[0].tolist()
                
                detections.append({
                    "part": PART_CLASSES[cls_id] if cls_id < len(PART_CLASSES) else f"class_{cls_id}",
                    "bbox": bbox,
                    "confidence": conf
                })
        
        return detections
    
    def predict_segmentation(self, img_tensor):
        """세영 세그멘테이션"""
        with torch.no_grad():
            out = self.seg_model(img_tensor)
            mask = out.argmax(dim=1).squeeze()
        return mask
    
    def predict_repair(self, img_tensor, mask_tensor):
        """세영 수리 분류"""
        combined = torch.cat([img_tensor, mask_tensor], dim=1)
        
        with torch.no_grad():
            out = self.rep_model(combined)
            probs = torch.sigmoid(out).squeeze()
        
        methods = []
        confs = {}
        
        for method, prob in zip(self.repair_methods, probs):
            p = prob.item()
            confs[method] = round(p, 4)
            if p >= self.threshold:
                methods.append(method)
        
        if not methods:
            idx = probs.argmax().item()
            methods.append(self.repair_methods[idx])
        
        return methods, confs
    
    def process_crop(self, crop_img):
        """크롭 이미지 처리"""
        img_tensor = self.transform(crop_img).unsqueeze(0).to(self.device)
        
        # 세그멘테이션
        mask = self.predict_segmentation(img_tensor)
        mask_np = mask.cpu().numpy()
        
        # 대미지 타입
        unique, counts = np.unique(mask_np[mask_np > 0], return_counts=True)
        if len(unique) > 0:
            damage_type = self.idx_to_damage.get(unique[counts.argmax()], "Unknown")
        else:
            damage_type = "No damage"
        
        # 면적 비율
        area_ratio = (mask_np > 0).sum() / mask_np.size
        
        # 수리 분류
        binary_mask = (mask > 0).float().unsqueeze(0).unsqueeze(0)
        methods, confs = self.predict_repair(img_tensor, binary_mask)
        
        return {
            "damage_type": damage_type,
            "damage_area_ratio": round(float(area_ratio), 4),
            "repair_methods": methods,
            "confidences": confs
        }
    
    def analyze(self, img_path):
        """전체 분석"""
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        
        # 문경 YOLO
        detections = self.detect_parts(img_path)
        
        # 각 부위별 세영 모델 적용
        items = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue
            
            crop = img.crop((x1, y1, x2, y2))
            result = self.process_crop(crop)
            
            items.append({
                "part": det["part"],
                "part_confidence": round(det["confidence"], 4),
                "damage_type": result["damage_type"],
                "damage_area_ratio": result["damage_area_ratio"],
                "repair_methods": result["repair_methods"],
                "confidences": result["confidences"],
                "bbox": [round(x, 2) for x in det["bbox"]]
            })
        
        return {
            "image_id": os.path.basename(img_path),
            "repair_items": items
        }


# ===================
# API
# ===================

_pipeline = None

def analyze_car(image_path):
    """
    메인 함수
    
    Args:
        image_path: 차량 이미지 경로
    
    Returns:
        {"image_id": "...", "repair_items": [...]}
    """
    global _pipeline
    if _pipeline is None:
        _pipeline = FullPipeline()
    return _pipeline.analyze(image_path)


# ===================
# 테스트
# ===================

if __name__ == "__main__":
    print("=" * 50)
    print("Full Pipeline Test")
    print("=" * 50)
    
    try:
        pipeline = FullPipeline()
    except Exception as e:
        print(f"[ERROR] {e}")
        exit(1)
    
    test_path = r"C:\Users\swu\Desktop\damage_sample\images"
    
    if os.path.isdir(test_path):
        imgs = [f for f in os.listdir(test_path) if f.endswith(('.jpg', '.png'))]
        if imgs:
            test_path = os.path.join(test_path, imgs[0])
    
    if os.path.exists(test_path):
        print(f"\nInput: {test_path}")
        result = pipeline.analyze(test_path)
        print("\nOutput:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"No test image: {test_path}")

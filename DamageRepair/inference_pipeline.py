"""
차량 파손 분석 파이프라인
- 1단계: U-Net 세그멘테이션 (대미지 위치와 면적을 파악)
- 2단계: ResNet34 분류 (수리 방법, Multi-Label = 여러 수리 방법 출력 가능)
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

# 모델 경로 (이 부분은 본인 환경에 맞게 수정하면 됨)
SEGMENTATION_MODEL_PATH = r"model\best_segmentation_model.pth"
SEGMENTATION_CLASSES_PATH = r"model\segmentation_classes.json"
REPAIR_MODEL_PATH = r"model\best_repair_model.pth"
REPAIR_CLASSES_PATH = r"model\repair_classes.json"

# 문경 YOLO 가져와서 사용
YOLO_MODEL_PATH = r"Yolo\model\best.pt"

# 부위 클래스 정의
PART_CLASSES = [
    "Front bumper", "Head lights", "Bonnet", "Windshield", "Roof",
    "Trunk lid", "Rear lamp", "Rear bumper", "Front fender", "Side mirror",
    "Front door", "Rear door", "Rocker panel", "A pillar", "B pillar",
    "C pillar", "Rear fender", "Front Wheel", "Rear Wheel", "Rear windshield",
    "Undercarriage"
]

# ===================
# 모델 정의
# ===================

class DoubleConv(nn.Module):
    """Conv-BN-ReLU 2번"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """U-Net 세그멘테이션"""
    def __init__(self, in_ch=3, num_classes=5):
        super().__init__()
        # 인코더
        self.enc1 = DoubleConv(in_ch, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024)
        
        # 디코더
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        
        self.out = nn.Conv2d(64, num_classes, 1)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        # 인코더
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        
        # 디코더 + skip connection
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        
        return self.out(d1)


class RepairClassifier(nn.Module):
    """수리 방법 분류 (4채널 입력, Multi-Label)"""
    def __init__(self, num_classes):
        super().__init__()
        resnet = models.resnet34(weights=None)
        
        # 4채널 입력 (RGB + mask)
        self.conv1 = nn.Conv2d(4, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        # 분류 헤드
        self.fc = nn.Sequential(
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
        return self.fc(x)


# ===================
# 파이프라인
# ===================

class DamageRepairPipeline:
    """대미지 분석 파이프라인"""
    
    def __init__(self, seg_path=SEGMENTATION_MODEL_PATH, seg_cls_path=SEGMENTATION_CLASSES_PATH,
                 rep_path=REPAIR_MODEL_PATH, rep_cls_path=REPAIR_CLASSES_PATH):
        
        self.device = DEVICE
        self.threshold = THRESHOLD
        
        # 전처리
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 세그멘테이션 모델
        print("Loading segmentation model...")
        with open(seg_cls_path, 'r', encoding='utf-8') as f:
            seg_info = json.load(f)
        
        self.damage_types = seg_info["damage_types"]
        self.seg_classes = seg_info["num_classes"]
        self.idx_to_damage = {v: k for k, v in seg_info["class_mapping"].items()}
        self.idx_to_damage[0] = "Background"
        
        self.seg_model = UNet(in_ch=3, num_classes=self.seg_classes)
        self.seg_model.load_state_dict(torch.load(seg_path, map_location=self.device))
        self.seg_model.to(self.device)
        self.seg_model.eval()
        print(f"  [OK] U-Net ({self.seg_classes} classes)")
        
        # 수리 분류 모델
        print("Loading repair classifier...")
        with open(rep_cls_path, 'r', encoding='utf-8') as f:
            rep_info = json.load(f)
        
        self.repair_methods = rep_info["repair_methods"]
        
        self.rep_model = RepairClassifier(num_classes=len(self.repair_methods))
        self.rep_model.load_state_dict(torch.load(rep_path, map_location=self.device))
        self.rep_model.to(self.device)
        self.rep_model.eval()
        print(f"  [OK] RepairClassifier ({len(self.repair_methods)} classes)")
    
    def predict_segmentation(self, img_tensor):
        """세그멘테이션 예측"""
        with torch.no_grad():
            out = self.seg_model(img_tensor)
            mask = out.argmax(dim=1).squeeze()
        return mask
    
    def predict_repair(self, img_tensor, mask_tensor):
        """수리 방법 예측 (Multi-Label)"""
        combined = torch.cat([img_tensor, mask_tensor], dim=1)
        
        with torch.no_grad():
            out = self.rep_model(combined)
            probs = torch.sigmoid(out).squeeze()
        
        # threshold 이상인 것 선택
        methods = []
        confs = {}
        
        for method, prob in zip(self.repair_methods, probs):
            p = prob.item()
            confs[method] = round(p, 4)
            if p >= self.threshold:
                methods.append(method)
        
        # 하나도 없으면 최댓값
        if not methods:
            idx = probs.argmax().item()
            methods.append(self.repair_methods[idx])
        
        return methods, confs
    
    def process_crop(self, crop_img):
        """크롭 이미지 처리"""
        img_tensor = self.transform(crop_img).unsqueeze(0).to(self.device)
        
        # 1단계: 세그멘테이션
        mask = self.predict_segmentation(img_tensor)
        mask_np = mask.cpu().numpy()
        
        # 대미지 타입 (가장 많은 클래스)
        unique, counts = np.unique(mask_np[mask_np > 0], return_counts=True)
        if len(unique) > 0:
            damage_type = self.idx_to_damage.get(unique[counts.argmax()], "Unknown")
        else:
            damage_type = "No damage"
        
        # 면적 비율
        area_ratio = (mask_np > 0).sum() / mask_np.size
        
        # 2단계: 수리 방법
        binary_mask = (mask > 0).float().unsqueeze(0).unsqueeze(0)
        methods, confs = self.predict_repair(img_tensor, binary_mask)
        
        return {
            "damage_type": damage_type,
            "damage_area_ratio": round(area_ratio, 4),
            "repair_methods": methods,
            "confidences": confs
        }
    
    def process_with_yolo(self, img_path, yolo_detections):
        """YOLO 결과로 전체 처리"""
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        
        items = []
        for det in yolo_detections:
            x1, y1, x2, y2 = det["bbox"]
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue
            
            crop = img.crop((x1, y1, x2, y2))
            result = self.process_crop(crop)
            
            items.append({
                "part": det["part"],
                "part_confidence": round(det.get("confidence", 0.9), 4),
                "damage_type": result["damage_type"],
                "damage_area_ratio": result["damage_area_ratio"],
                "repair_methods": result["repair_methods"],
                "confidences": result["confidences"],
                "bbox": det["bbox"]
            })
        
        return {"image_id": os.path.basename(img_path), "repair_items": items}


# ===================
# API
# ===================

_pipeline = None

def analyze_damage(image_path, yolo_detections):
    """
    메인 함수
    
    Args:
        image_path: 이미지 경로
        yolo_detections: [{"part": "...", "bbox": [x1,y1,x2,y2], "confidence": 0.9}, ...]
    
    Returns:
        {"image_id": "...", "repair_items": [...]}
    """
    global _pipeline
    if _pipeline is None:
        _pipeline = DamageRepairPipeline()
    return _pipeline.process_with_yolo(image_path, yolo_detections)


# ===================
# 테스트
# ===================

if __name__ == "__main__":
    print("=" * 50)
    print("Damage Analysis Pipeline")
    print("=" * 50)
    
    try:
        pipeline = DamageRepairPipeline()
    except Exception as e:
        print(f"[ERROR] {e}")
        exit(1)
    
    # YOLO 로드
    yolo = None
    if os.path.exists(YOLO_MODEL_PATH):
        try:
            from ultralytics import YOLO
            yolo = YOLO(YOLO_MODEL_PATH)
            print("[OK] YOLO loaded")
        except:
            print("[WARN] YOLO load failed")
    
    # 테스트
    test_path = r"C:\Users\swu\Desktop\damage_sample\images"
    
    if os.path.isdir(test_path):
        imgs = [f for f in os.listdir(test_path) if f.endswith(('.jpg', '.png'))]
        if imgs:
            test_path = os.path.join(test_path, imgs[0])
    
    if os.path.exists(test_path):
        print(f"\nInput: {test_path}")
        
        if yolo:
            results = yolo(test_path)
            dets = []
            for r in results:
                for box in r.boxes:
                    dets.append({
                        "part": PART_CLASSES[int(box.cls[0])],
                        "bbox": box.xyxy[0].tolist(),
                        "confidence": float(box.conf[0])
                    })
        else:
            # 더미
            dets = [{"part": "Front bumper", "bbox": [100, 150, 400, 350], "confidence": 0.9}]
        
        out = pipeline.process_with_yolo(test_path, dets)
        print("\nOutput:")
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(f"\nNo test image: {test_path}")

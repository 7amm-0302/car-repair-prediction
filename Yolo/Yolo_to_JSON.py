import glob
import os
import json
from ultralytics import YOLO

part_model = YOLO(r"C:\Users\swu\Desktop\guaze\runs\detect\final3\weights\best.pt")

def run_part_detector(image_path: str):
    results = part_model(image_path, imgsz=640, conf=0.25)
    r = results[0]

    boxes = r.boxes.xyxy.cpu().numpy()
    cls_ids = r.boxes.cls.cpu().numpy().astype(int)
    scores = r.boxes.conf.cpu().numpy()
    names = part_model.names

    parts = []
    for i, (xyxy, cid, score) in enumerate(zip(boxes, cls_ids, scores), start=1):
        x1, y1, x2, y2 = xyxy.tolist()
        parts.append({
            "id": i,
            "part": names[cid],
            "class_id": int(cid),
            "bbox": [x1, y1, x2, y2],
            "score": float(score),
        })

    return {"image_path": image_path, "parts": parts}

if __name__ == "__main__":
    image_root = r"C:\Users\swu\Desktop\sample\part_sample_17190\1.원천데이터"

    # 1) 이미지 확장자/하위폴더까지 다 찾기
    patterns = ["*.jpg", "*.JPG", "*.png", "*.jpeg", "*.JPEG"]
    img_paths = []
    for p in patterns:
        img_paths.extend(
            glob.glob(os.path.join(image_root, "**", p), recursive=True)
        )

    print("찾은 이미지 개수:", len(img_paths))
    if img_paths:
        print("예시 3개:", img_paths[:3])

    # 2) 없으면 바로 종료
    if not img_paths:
        print("이미지를 하나도 못 찾았음. image_root / 확장자 확인 필요!")
    else:
        for img_path in img_paths:
            analysis = run_part_detector(img_path)

            json_path = os.path.splitext(img_path)[0] + "_part.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2)

            print("saved:", json_path)

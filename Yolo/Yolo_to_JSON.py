import glob
import os
import json
from ultralytics import YOLO
from tqdm import tqdm

# 모델 로드
part_model = YOLO(r"C:\Users\swu\Desktop\guaze\runs\detect\final4\weights\best.pt")

# ==========================================
# ★ 여기만 수정하세요 (JSON 저장할 폴더 경로)
# ==========================================
SAVE_ROOT = r"C:\Users\swu\Desktop\sample\part_sample_10000"

# 설정값
CONF_THRES = 0.15
IOU_THRES = 0.5

# 저장 폴더가 없으면 자동으로 생성 (에러 방지)
os.makedirs(SAVE_ROOT, exist_ok=True)


def run_part_detector(image_path: str):
    results = part_model.predict(
        source=image_path,
        imgsz=640,
        conf=CONF_THRES,
        iou=IOU_THRES,
        verbose=False
    )
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
            "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            "score": float(round(score, 4)),
        })

    return {
        "image_path": image_path,
        "image_size": {"width": r.orig_shape[1], "height": r.orig_shape[0]},
        "parts": parts
    }


if __name__ == "__main__":
    image_root = r"C:\Users\swu\Desktop\sample\part_sample_10000\1.원천데이터"

    print("🔍 이미지 검색 중...")
    patterns = ["*.jpg"]
    img_paths = []
    for p in patterns:
        img_paths.extend(
            glob.glob(os.path.join(image_root, "**", p), recursive=True)
        )

    print(f"✅ 총 {len(img_paths)}개의 이미지를 찾았습니다.")

    if not img_paths:
        print("❌ 이미지를 하나도 못 찾았습니다.")
    else:
        print(f"🚀 분석 시작! 결과는 [{SAVE_ROOT}] 폴더에 저장됩니다.")

        for img_path in tqdm(img_paths, desc="Processing"):

            # 1. 파일 이름만 따오기 (예: "car_01.jpg")
            filename = os.path.basename(img_path)

            # 2. 확장자 떼고 "_part.json" 붙이기 (예: "car_01_part.json")
            json_filename = os.path.splitext(filename)[0] + "_part.json"

            # 3. 최종 저장 경로 만들기 (SAVE_ROOT + 파일명)
            save_path = os.path.join(SAVE_ROOT, json_filename)

            # (이어하기 기능) 이미 있으면 건너뛰기
            if os.path.exists(save_path):
                continue

            analysis = run_part_detector(img_path)

            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2)

        print("🎉 모든 분석 완료!")
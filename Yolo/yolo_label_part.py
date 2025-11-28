import os
import json

# ===== 경로만 너 PC에 맞게 수정 =====
PART_SAMPLE_ROOT = r"C:\Users\swu\Desktop\sample\part_sample_17190"

IMG_DIR  = os.path.join(PART_SAMPLE_ROOT, r"1.원천데이터", "part")   # 이미지 폴더
JSON_DIR = os.path.join(PART_SAMPLE_ROOT, r"2.라벨링데이터", "part")  # part 라벨 JSON 폴더
LBL_DIR  = os.path.join(PART_SAMPLE_ROOT, "yolo_labels_part_bbox_merged")   # YOLO 라벨 저장 폴더

os.makedirs(LBL_DIR, exist_ok=True)

MERGED_CLASSES = [
    "Front bumper",   # 0
    "Head lights",    # 1
    "Bonnet",         # 2
    "Windshield",     # 3
    "Roof",           # 4
    "Trunk lid",      # 5
    "Rear lamp",      # 6
    "Rear bumper",    # 7
    "Front fender",   # 8
    "Side mirror",    # 9
    "Front door",     # 10
    "Rear door",      # 11
    "Rocker panel",   # 12
    "A pillar",       # 13
    "B pillar",       # 14
    "C pillar",       # 15
    "Rear fender",    # 16
    "Front Wheel",    # 17
    "Rear Wheel",     # 18
    "Rear windshield",# 19
    "Undercarriage",  # 20
]

MERGED_PART_MAP = {name: idx for idx, name in enumerate(MERGED_CLASSES)}

NORMALIZE_NAME = {
    "Front bumper": "Front bumper",
    "Bonnet": "Bonnet",
    "Windshield": "Windshield",
    "Roof": "Roof",
    "Trunk lid": "Trunk lid",
    "Rear bumper": "Rear bumper",
    "Rear windshield": "Rear windshield",
    "Undercarriage": "Undercarriage",

    # 좌/우 합치기
    "Head lights(L)": "Head lights",
    "Head lights(R)": "Head lights",

    "Rear lamp(L)": "Rear lamp",
    "Rear lamp(R)": "Rear lamp",

    "Front fender(L)": "Front fender",
    "Front fender(R)": "Front fender",

    "Side mirror(L)": "Side mirror",
    "Side mirror(R)": "Side mirror",

    "Front door(L)": "Front door",
    "Front door(R)": "Front door",

    "Rear door(L)": "Rear door",
    "Rear door(R)": "Rear door",

    "Rocker panel(L)": "Rocker panel",
    "Rocker panel(R)": "Rocker panel",

    "A pillar(L)": "A pillar",
    "A pillar(R)": "A pillar",

    "B pillar(L)": "B pillar",
    "B pillar(R)": "B pillar",

    "C pillar(L)": "C pillar",
    "C pillar(R)": "C pillar",

    "Rear fender(L)": "Rear fender",
    "Rear fender(R)": "Rear fender",

    "Front Wheel(L)": "Front Wheel",
    "Front Wheel(R)": "Front Wheel",

    "Rear Wheel(L)": "Rear Wheel",
    "Rear Wheel(R)": "Rear Wheel",
}


def json_to_yolo_for_one_image(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        j = json.load(f)

    file_name = j["images"]["file_name"]
    W = j["images"]["width"]
    H = j["images"]["height"]

    labels = []

    for anno in j.get("annotations", []):
        raw_name = anno.get("part")
        if not raw_name:
            continue

        # 1) 원래 이름을 통합 이름으로 변환
        norm_name = NORMALIZE_NAME.get(raw_name)
        if norm_name is None:
            continue

        # 2) 통합 이름을 class id로 변환
        if norm_name not in MERGED_PART_MAP:
            continue

        cls_id = MERGED_PART_MAP[norm_name]

        # COCO 형식 bbox: [x_min, y_min, w, h]
        x_min, y_min, bw, bh = anno["bbox"]

        x_center = (x_min + bw / 2.0) / W
        y_center = (y_min + bh / 2.0) / H
        w_norm = bw / W
        h_norm = bh / H

        labels.append((cls_id, x_center, y_center, w_norm, h_norm))

    return file_name, labels


def main():
    json_files = [f for f in os.listdir(JSON_DIR) if f.lower().endswith(".json")]
    print("part JSON 개수:", len(json_files))

    for jf in json_files:
        stem, _ = os.path.splitext(jf)
        json_path = os.path.join(JSON_DIR, jf)

        file_name, labels = json_to_yolo_for_one_image(json_path)
        img_path = os.path.join(IMG_DIR, file_name)

        if not os.path.exists(img_path):
            print("[경고] 이미지 없음:", img_path)
            continue

        out_txt = os.path.join(LBL_DIR, stem + ".txt")
        with open(out_txt, "w", encoding="utf-8") as f:
            for cls_id, xc, yc, w_norm, h_norm in labels:
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w_norm:.6f} {h_norm:.6f}\n")

    print("YOLO bbox 라벨 (좌/우 통합 버전) 변환 완료:", LBL_DIR)


if __name__ == "__main__":
    main()

import os
import json
import numpy as np
import cv2
import pandas as pd
from PIL import Image

DAMAGE_MAP = {
    "Scratched": 1,
    "Crushed": 2,
    "Breakage": 3,
    "Separated": 4,
}

PART_MAP = {
    "Front bumper": 1,
    "Head lights(L)": 2,
    "Head lights(R)": 3,
    "Bonnet": 4,
    "Windshield": 5,
    "Roof": 6,
    "Trunk lid": 7,
    "Rear lamp(L)": 8,
    "Rear lamp(R)": 9,
    "Rear bumper": 10,
    "Front fender(L)": 11,
    "Front fender(R)": 12,
    "Side mirror(L)": 13,
    "Side mirror(R)": 14,
    "Front door(L)": 15,
    "Front door(R)": 16,
    "Rear door(L)": 17,
    "Rear door(R)": 18,
    "Rocker panel(L)": 19,
    "Rocker panel(R)": 20,
    "A pillar(L)": 21,
    "A pillar(R)": 22,
    "B pillar(L)": 23,
    "B pillar(R)": 24,
    "C pillar(L)": 25,
    "C pillar(R)": 26,
    "Rear fender(L)": 27,
    "Rear fender(R)": 28,
    "Front Wheel(L)": 29,
    "Front Wheel(R)": 30,
    "Rear Wheel(L)": 31,
    "Rear Wheel(R)": 32,
    "Rear windshield": 37,
    "Undercarriage": 38,
}

def make_mask(json_path, class_key, class_map, out_path):
    with open(json_path, "r", encoding="utf-8") as f:
        j = json.load(f)

    H = j["images"]["height"]
    W = j["images"]["width"]
    mask = np.zeros((H, W), dtype=np.uint8)

    for anno in j["annotations"]:
        cls_name = anno.get(class_key)
        if not cls_name:
            continue

        cls = class_map.get(cls_name)
        if cls is None:
            print(f"[경고] 매핑 안된 {class_key}: {cls_name} ({json_path})")
            continue

        for poly_group in anno["segmentation"]:
            for poly in poly_group:
                pts = np.array(poly, dtype=np.int32)
                cv2.fillPoly(mask, [pts], int(cls))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(mask).save(out_path)


DATASETS = [
    {
        "name": "damage",
        "root": r"C:\Users\swu\Desktop\sample\damage_sample_5000",
        "index_csv": r"C:\Users\swu\Desktop\sample\damage_sample_5000\index_damage.csv",
    },
    {
        "name": "part",
        "root": r"C:\Users\swu\Desktop\sample\part_sample_5000",
        "index_csv": r"C:\Users\swu\Desktop\sample\part_sample_5000\index_part.csv",
    },
]

for ds in DATASETS:
    name = ds["name"]
    root = ds["root"]
    index_csv = ds["index_csv"]

    print(f"\n=== {name} 세트 마스크 생성 시작 ===")
    print("root     :", root)
    print("index_csv:", index_csv)

    df = pd.read_csv(index_csv)
    print("행 수:", len(df))

    out_dir_damage = os.path.join(root, "masks_damage")
    out_dir_part   = os.path.join(root, "masks_part")

    cnt_damage = 0
    cnt_part = 0

    for _, row in df.iterrows():
        fname = row["file_name"]
        stem, _ = os.path.splitext(fname)

        # damage json 있으면 damage 마스크
        dmg_path = row.get("damage_json_path", "")
        if isinstance(dmg_path, str) and dmg_path.strip() != "":
            out_path = os.path.join(out_dir_damage, stem + ".png")
            make_mask(dmg_path, "damage", DAMAGE_MAP, out_path)
            cnt_damage += 1

        # part json 있으면 part 마스크
        part_path = row.get("part_json_path", "")
        if isinstance(part_path, str) and part_path.strip() != "":
            out_path = os.path.join(out_dir_part, stem + ".png")
            make_mask(part_path, "part", PART_MAP, out_path)
            cnt_part += 1

    print(f"{name} 세트: damage 마스크 {cnt_damage}개, part 마스크 {cnt_part}개")
    print("  damage masks:", out_dir_damage)
    print("  part   masks:", out_dir_part)

import os
import glob
import json
import re
import pandas as pd

# --------------------------------------------------
# 1. 경로 설정
# --------------------------------------------------

TRAIN_ROOT = r"C:\Users\swu\Desktop\160._차량파손_이미지_데이터\01.데이터\1.Training"

DAMAGE_JSON_DIR = os.path.join(TRAIN_ROOT, "2.라벨링데이터", "damage")
PART_JSON_DIR = os.path.join(TRAIN_ROOT, "2.라벨링데이터", "damage_part")
EST_DIR = r"C:\Users\swu\Desktop\160._차량파손_이미지_데이터\01.데이터\1.Training\3.견적서데이터\TS_99_견적서"

print("damage 라벨 폴더:", DAMAGE_JSON_DIR)
print("part 라벨 폴더  :", PART_JSON_DIR)
print("견적서 폴더     :", EST_DIR)


# --------------------------------------------------
# 2. 헬퍼 함수들
# --------------------------------------------------

def parse_total_cost(j_est):
    try:
        total_str = j_est["수리비 정산정보"]["합계"]["총계"]
    except KeyError:
        return None
    digits = re.sub(r"[^\d]", "", total_str)
    return int(digits) if digits else None


def parse_car_type(j_est):
    car_info = j_est.get("차량정보", {}) or {}
    car_type = car_info.get("제작사/차종")
    if car_type:
        return car_type
    maker = car_info.get("제작사") or ""
    name = car_info.get("차량명칭") or car_info.get("차종") or ""
    combo = f"{maker} {name}".strip()
    return combo if combo else None


def parse_repair_raw(json_path):

    if not os.path.exists(json_path):
        return None, []

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            j = json.load(f)
    except Exception as e:
        print(f"[Error] JSON 로드 실패: {json_path} / {e}")
        return None, []

    file_name = j.get("images", {}).get("file_name")
    repair_list = []

    for anno in j.get("annotations", []):
        repairs_field = anno.get("repair")
        if not repairs_field:
            continue
        if isinstance(repairs_field, str):
            repairs_field = [repairs_field]
        for item in repairs_field:
            if item:
                repair_list.append(str(item).strip())

    return file_name, repair_list


# --------------------------------------------------
# 3. 메인 로직
# --------------------------------------------------

rows = []

dmg_files = set(os.path.basename(p) for p in glob.glob(os.path.join(DAMAGE_JSON_DIR, "*.json")))
part_files = set(os.path.basename(p) for p in glob.glob(os.path.join(PART_JSON_DIR, "*.json")))

all_files = dmg_files | part_files

print(f"Damage 폴더 파일 수: {len(dmg_files)}")
print(f"Part 폴더 파일 수  : {len(part_files)}")


for i, filename in enumerate(all_files):

    stem = os.path.splitext(filename)[0]
    if "_" not in stem:
        continue

    image_code, accident_id = stem.split("_", 1)

    est_path = os.path.join(EST_DIR, accident_id + ".json")
    if not os.path.exists(est_path):
        continue

    dmg_path_full = os.path.join(DAMAGE_JSON_DIR, filename)
    part_path_full = os.path.join(PART_JSON_DIR, filename)

    fname_d, repairs_d = parse_repair_raw(dmg_path_full)
    fname_p, repairs_p = parse_repair_raw(part_path_full)

    final_file_name = fname_d if fname_d else fname_p
    if not final_file_name:
        final_file_name = stem + ".jpg"

    combined_repairs = list(set(repairs_d + repairs_p))

    if not combined_repairs:
        continue
-
    try:
        with open(est_path, "r", encoding="utf-8") as f:
            j_est = json.load(f)
    except Exception:
        continue

    total_cost = parse_total_cost(j_est)
    car_type = parse_car_type(j_est)

    rows.append({
        "file_name": final_file_name,
        "image_code": image_code,
        "accident_id": accident_id,
        "repair": "|".join(combined_repairs),  # 구분자 | 로 연결
        "car_type": car_type,
        "total_cost": total_cost,
    })

print("-" * 30)

# --------------------------------------------------
# 4. 저장
# --------------------------------------------------

df = pd.DataFrame(rows)
print("매칭 성공 행 수:", len(df))

df = df[df["total_cost"].notna()].copy()
print("최종 유효 데이터 수:", len(df))

out_csv = os.path.join(TRAIN_ROOT, "labels_repair_cost_12.csv")
df.to_csv(out_csv, index=False, encoding="utf-8-sig")

print("CSV 저장 완료:", out_csv)
print(df.head())
import pandas as pd
import numpy as np

df = pd.read_csv("labels_repair_cost_12.csv")


# ======================
# 1) repair 파싱 함수
# ======================
def parse_repair(repair_str):
    parts = {}
    for seg in repair_str.split('|'):
        if ':' not in seg:
            continue
        part, worklist = seg.split(':')
        works = worklist.split(',')
        if part not in parts:
            parts[part] = set()
        parts[part].update(works)  # 작업 목록 union
    return parts


# ======================
# 2) 병합된 repair 문자열 생성 함수
# ======================
def merge_repair_strings(repair_list):
    merged = {}

    for rep in repair_list:
        parts_dict = parse_repair(rep)
        for part, works in parts_dict.items():
            if part not in merged:
                merged[part] = set()
            merged[part].update(works)

    merged_str = '|'.join([
        f"{part}:{','.join(sorted(works))}"
        for part, works in merged.items()
    ])
    return merged_str


# ======================
# 3) accident_id 기준 병합
# ======================
def merge_duplicates(df):
    merged_rows = []

    for acc_id, group in df.groupby('accident_id'):
        if len(group) == 1:
            merged_rows.append(group.iloc[0])
            continue

        # 1) repair 병합
        merged_repair = merge_repair_strings(group['repair'].tolist())

        # 2) file_name 병합 → 중복 제거 + 정렬
        merged_file_name = "|".join(sorted(set(group['file_name'].astype(str))))

        # 3) image_code 병합 → 중복 제거 + 정렬
        merged_image_code = "|".join(sorted(set(group['image_code'].astype(str))))

        # 대표 row 하나 가져오기
        row = group.iloc[0].copy()
        row['repair'] = merged_repair
        row['file_name'] = merged_file_name
        row['image_code'] = merged_image_code

        merged_rows.append(row)

    df_merged = pd.DataFrame(merged_rows)
    return df_merged.reset_index(drop=True)


# ======================
# 4) 실행
# ======================
df_clean = merge_duplicates(df)

output_path = "labels_repair_cost_12_clean.csv"
df_clean.to_csv(output_path, index=False, encoding="utf-8-sig")

print("저장 완료:", output_path)
print("행 개수:", len(df_clean))

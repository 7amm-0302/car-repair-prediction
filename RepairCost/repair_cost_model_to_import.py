import json
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from xgboost import XGBRegressor


# -------------------------------
# Canonical Mapping (훈련과 동일)
# -------------------------------
PART_CANONICAL = {
    "Front bumper": "Front bumper",
    "Rear bumper": "Rear bumper",

    "Head lights(L)": "Head lights",
    "Head lights(R)": "Head lights",
    "Head lights": "Head lights",

    "Bonnet": "Bonnet",
    "Windshield": "Windshield",
    "Roof": "Roof",
    "Trunk lid": "Trunk lid",

    "Rear lamp(L)": "Rear lamp",
    "Rear lamp(R)": "Rear lamp",
    "Rear lamp": "Rear lamp",

    "Front fender(L)": "Front fender",
    "Front fender(R)": "Front fender",
    "Front fender": "Front fender",

    "Side mirror(L)": "Side mirror",
    "Side mirror(R)": "Side mirror",
    "Side mirror": "Side mirror",

    "Front door(L)": "Front door",
    "Front door(R)": "Front door",
    "Front door": "Front door",

    "Rear door(L)": "Rear door",
    "Rear door(R)": "Rear door",
    "Rear door": "Rear door",

    "Rocker panel(L)": "Rocker panel",
    "Rocker panel(R)": "Rocker panel",
    "Rocker panel": "Rocker panel",

    "A pillar(L)": "A pillar",
    "A pillar(R)": "A pillar",
    "A pillar": "A pillar",

    "B pillar(L)": "B pillar",
    "B pillar(R)": "B pillar",
    "B pillar": "B pillar",

    "C pillar(L)": "C pillar",
    "C pillar(R)": "C pillar",
    "C pillar": "C pillar",

    "Rear fender(L)": "Rear fender",
    "Rear fender(R)": "Rear fender",
    "Rear fender": "Rear fender",

    "Front Wheel(L)": "Front Wheel",
    "Front Wheel(R)": "Front Wheel",
    "Front Wheel": "Front Wheel",

    "Rear Wheel(L)": "Rear Wheel",
    "Rear Wheel(R)": "Rear Wheel",
    "Rear Wheel": "Rear Wheel",

    "Rear windshield": "Rear windshield",
    "Undercarriage": "Undercarriage",
}


class RepairCostPipeline:
    def __init__(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        
        # 모델 불러오기
        self.model = XGBRegressor()
        self.model.load_model(os.path.join(BASE_DIR, "model/repair_cost_model.json"))

        # 멀티핫 인코더 불러오기
        with open(os.path.join(BASE_DIR, "model/mlb_parts.pkl"), "rb") as f:
            self.mlb_parts = pickle.load(f)

        with open(os.path.join(BASE_DIR, "model/mlb_works.pkl"), "rb") as f:
            self.mlb_works = pickle.load(f)

        with open(os.path.join(BASE_DIR, "model/mlb_partwork.pkl"), "rb") as f:
            self.mlb_partwork = pickle.load(f)

        # 차종 원핫 컬럼 불러오기
        with open(os.path.join(BASE_DIR, "model/car_columns.json"), "r", encoding="utf-8") as f:
            self.car_columns = json.load(f)

    def canonicalize_part(self, part):
        return PART_CANONICAL.get(part, part)

    def parse_repair(self, repair_str):
        if pd.isna(repair_str):
            return [], [], []

        parts = []
        works_total = []
        partwork_pairs = []

        for seg in repair_str.split("|"):
            if ":" not in seg:
                continue

            raw_part, work_str = seg.split(":", 1)
            part = self.canonicalize_part(raw_part)
            works = work_str.split(",")

            parts.append(part)
            works_total.extend(works)

            for w in works:
                partwork_pairs.append(f"{part}_{w}")

        return parts, works_total, partwork_pairs

    def transform_row(self, row):
        parts, works, partwork = self.parse_repair(row["repair_pred"])

        X_part = self.mlb_parts.transform([parts])[0]
        X_work = self.mlb_works.transform([works])[0]
        X_pw   = self.mlb_partwork.transform([partwork])[0]

        # numeric feature
        num_parts = len(parts)
        num_works = len(works)
        num_sheet = works.count("sheet_metal")
        num_exchg = works.count("exchange")
        num_rep   = works.count("repair")
        X_num = np.array([num_parts, num_works, num_sheet, num_exchg, num_rep])

        # car_type 원핫
        X_car = np.zeros(len(self.car_columns))
        car = row["car_type"]
        if car in self.car_columns:
            idx = self.car_columns.index(car)
            X_car[idx] = 1

        return np.hstack([X_pw, X_part, X_num, X_car])

    def predict_df(self, df_pred):
        valid_rows = []
        valid_idx = []
        skipped_idx = []

        def contains_unknown(parts, works, pws, known_parts, known_works, known_pw):
            return (
                any(p not in known_parts for p in parts) or
                any(w not in known_works for w in works) or
                any(pw not in known_pw for pw in pws)
            )

        known_parts = set(self.mlb_parts.classes_)
        known_works = set(self.mlb_works.classes_)
        known_pw    = set(self.mlb_partwork.classes_)

        # -------------------
        # 1) unknown 제거
        # -------------------
        for idx, row in df_pred.iterrows():
            parts, works, partwork = self.parse_repair(row["repair_pred"])

            if contains_unknown(parts, works, partwork,
                                known_parts, known_works, known_pw):
                skipped_idx.append(idx)
            else:
                valid_idx.append(idx)
                valid_rows.append(row)

        if len(valid_rows) == 0:
            print("⚠ 모든 샘플 제외됨 → 예측 불가")
            df_pred["pred_cost"] = np.nan
            return df_pred

        # valid 한 row만 추출
        df_valid = df_pred.loc[valid_idx].copy()

        # -------------------
        # 2) feature 생성
        # -------------------
        X_list = [self.transform_row(row) for row in valid_rows]
        X_arr = np.vstack(X_list)

        pred_log = self.model.predict(X_arr)
        pred_cost = np.expm1(pred_log).astype(int)

        df_valid["pred_cost"] = pred_cost

        print(f"Unknown 제외: {len(skipped_idx)}개")

        return df_valid



def run_cost_prediction(df_pred):
    pipeline = RepairCostPipeline()
    df_pred = pipeline.predict_df(df_pred)

    df_pred["pred_cost"] = df_pred["pred_cost"].astype(int)
    return df_pred

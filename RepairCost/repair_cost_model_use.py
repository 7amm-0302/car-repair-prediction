import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from xgboost import XGBRegressor


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
    def __init__(self, model_json_path):
        self.model = XGBRegressor()
        self.model.load_model(model_json_path)

        self.mlb_parts = MultiLabelBinarizer()
        self.mlb_works = MultiLabelBinarizer()
        self.mlb_partwork = MultiLabelBinarizer()
        self.car_types = None

    def canonicalize_part(self, part):
        return PART_CANONICAL.get(part, part)

    def parse_repair(self, repair_str):
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

        # numeric
        num_parts = len(parts)
        num_works = len(works)

        num_sheet = works.count("sheet_metal")
        num_exchg = works.count("exchange")
        num_repair = works.count("repair")

        X_num = np.array([num_parts, num_works, num_sheet, num_exchg, num_repair])

        # car type one-hot manually
        X_car = np.zeros(len(self.car_types))
        if row["car_type"] in self.car_types:
            idx = self.car_types.index(row["car_type"])
            X_car[idx] = 1

        return np.hstack([X_pw, X_part, X_work, X_num, X_car])

    def predict_df(self, df_pred):
        X_list = []
        for _, row in df_pred.iterrows():
            X_list.append(self.transform_row(row))

        X_arr = np.vstack(X_list)
        pred_log = self.model.predict(X_arr)
        pred_cost = np.expm1(pred_log)
        return pred_cost


def run_cost_prediction_pipeline(model_json_path, df_pred):

    pipeline = RepairCostPipeline(model_json_path)

    pred_cost = pipeline.predict_df(df_pred)
    df_pred["pred_cost"] = pred_cost.astype(int)

    return df_pred

import os
import uuid
import pandas as pd
from flask import Flask, request, render_template_string

from tqdm import tqdm

from DamageRepair.inference_full import analyze_car
from RepairCost.repair_cost_model_to_import import run_cost_prediction

# --------------------
# 설정
# --------------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# --------------------
# repair 문자열 추출
# --------------------
def extract_repair_from_images(image_paths):
    """
    여러 장의 이미지에서 repair 문자열 추출해서
    최종 repair 문자열 하나로 합치는 함수.
    """
    all_items = []

    for img_path in image_paths:
        try:
            result = analyze_car(img_path)
        except Exception as e:
            print(f"[ERROR] analyze_car 실패: {img_path}, 이유: {e}")
            continue

        # result["repair_items"] 안에 파트별 정보가 들어있다고 가정
        for item in result.get("repair_items", []):
            part = item["part"]
            methods = item["repair_methods"]  # ["coating", "exchange", ...]
            all_items.append((part, methods))

    # 파트별로 방법들 합치기
    merged = {}
    for part, methods in all_items:
        if part not in merged:
            merged[part] = set()
        merged[part].update(methods)

    # "Front bumper:coating,exchange|Rear door:coating" 형식으로 직렬화
    repair_str_list = []
    for part, methods in merged.items():
        methods_str = ",".join(sorted(methods))
        repair_str_list.append(f"{part}:{methods_str}")

    repair_str = "|".join(repair_str_list)
    return repair_str


# --------------------
# df: file_name, accident_id, car_type, total_cost
# --------------------
def predict_cost(df):
    rows = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        file_names = row["file_name"].split("|")
        image_paths = [fn.strip() for fn in file_names]

        acc_id = row["accident_id"]
        car_type = row["car_type"]
        total_cost = row.get("total_cost", 0)  # 웹에서는 실제값 없으니 0으로 둠

        repair_pred = extract_repair_from_images(image_paths)

        rows.append({
            "accident_id": acc_id,
            "car_type": car_type,
            "repair_true": row.get("repair", ""),  # 웹에서는 공란
            "repair_pred": repair_pred,
            "total_cost": total_cost
        })

    df_pred = pd.DataFrame(rows)

    # run_cost_prediction은 df_pred에 pred_cost 컬럼을 붙여서 반환한다고 가정
    result = run_cost_prediction(df_pred)
    return result


# --------------------
# 템플릿 (간단히 render_template_string 사용)
# --------------------
PAGE_TEMPLATE = """
<!doctype html>
<html lang="ko">
  <head>
    <meta charset="utf-8">
    <title>자동차 수리비 예측 데모</title>
    <style>
      body { font-family: sans-serif; margin: 40px; }
      .container { max-width: 800px; margin: 0 auto; }
      .card { border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin-top: 20px; }
      .result { font-size: 1.2rem; font-weight: bold; color: #333; }
      label { display:block; margin-top: 10px; }
      input[type="text"] { width: 100%; padding: 6px; margin-top: 4px; }
      input[type="file"] { margin-top: 4px; }
      button { margin-top: 16px; padding: 8px 16px; }
      pre { background:#f7f7f7; padding:10px; border-radius:5px; }
    </style>
  </head>
  <body>
    <div class="container">
      <h1>자동차 이미지 기반 수리비 예측 데모</h1>

      <div class="card">
        <form method="post" enctype="multipart/form-data">
          <label>
            차종 (예: 현대, 기아 등)
            <input type="text" name="car_type" required>
          </label>

          <label>
            차량 이미지 (여러 장 선택 가능)
            <input type="file" name="images" multiple required>
          </label>

          <button type="submit">수리비 예측하기</button>
        </form>
      </div>

      {% if predicted %}
      <div class="card">
        <div class="result">
          예측 수리비: {{ predicted_cost | int }} 원
        </div>
        <p>추출된 repair 문자열:</p>
        <pre>{{ repair_pred }}</pre>
      </div>
      {% endif %}
    </div>
  </body>
</html>
"""


# --------------------
# 라우트
# --------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template_string(PAGE_TEMPLATE, predicted=False)

    # POST: 폼에서 받은 데이터 처리
    car_type = request.form.get("car_type", "").strip()
    files = request.files.getlist("images")

    if not car_type or not files:
        return render_template_string(PAGE_TEMPLATE, predicted=False)

    # 업로드된 파일 저장 + 경로 모으기
    saved_paths = []
    for f in files:
        if not f.filename:
            continue
        # 파일명 충돌 방지용 랜덤 ID
        ext = os.path.splitext(f.filename)[1]
        fname = f"{uuid.uuid4().hex}{ext}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
        f.save(save_path)
        saved_paths.append(save_path)

    if not saved_paths:
        return render_template_string(PAGE_TEMPLATE, predicted=False)

    # CSV 한 행 모양으로 DataFrame 구성
    # file_name 컬럼은 "path1|path2|..." 형식으로
    file_name_str = "|".join(saved_paths)
    temp_df = pd.DataFrame([{
        "file_name": file_name_str,
        "accident_id": f"web-{uuid.uuid4().hex}",  # 임의 ID
        "car_type": car_type,
        "total_cost": 0  # 웹에서는 실제값 없음
    }])

    # 수리비 예측
    result_df = predict_cost(temp_df)
    # result_df에는 pred_cost와 repair_pred가 포함되어 있다고 가정
    pred_cost = float(result_df.loc[0, "pred_cost"])
    repair_pred = result_df.loc[0, "repair_pred"]

    return render_template_string(
        PAGE_TEMPLATE,
        predicted=True,
        predicted_cost=pred_cost,
        repair_pred=repair_pred
    )


if __name__ == "__main__":
    # host='0.0.0.0' 으로 바꾸면 외부 접속도 가능
    app.run(debug=True)

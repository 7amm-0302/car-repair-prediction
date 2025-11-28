import os
import numpy as np
from PIL import Image

# ----- 경로 설정 -----
SAMPLE_ROOT = r"C:\Users\swu\Desktop\sample\damage_sample_5000"

IMG_DIR   = os.path.join(SAMPLE_ROOT, "1.원천데이터", "damage")
MASK_DIR  = os.path.join(SAMPLE_ROOT, "masks_damage")

OUT_COLOR_DIR   = os.path.join(SAMPLE_ROOT, "masks_damage_vis")        # 컬러 마스크
OUT_OVERLAY_DIR = os.path.join(SAMPLE_ROOT, "masks_damage_overlay")    # 원본+마스크 합성

os.makedirs(OUT_COLOR_DIR, exist_ok=True)
os.makedirs(OUT_OVERLAY_DIR, exist_ok=True)

# ----- damage 클래스별 색깔 지정  -----
# (R, G, B)
COLOR_MAP_DAMAGE = {
    0: (0,   0,   0),       # 배경
    1: (0,   255, 0),       # Scratched  -> 초록
    2: (255, 0,   0),       # Crushed    -> 빨강
    3: (0,   0,   255),     # Breakage   -> 파랑
    4: (255, 255, 0),       # Separated  -> 노랑
}

alpha = 0.5  # 오버레이 투명도 (0~1 사이)

for fname in os.listdir(MASK_DIR):
    if not fname.lower().endswith(".png"):
        continue

    mask_path = os.path.join(MASK_DIR, fname)
    mask = np.array(Image.open(mask_path))  # H x W (uint8)

    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    # 클래스별로 색 칠하기
    for label, rgb in COLOR_MAP_DAMAGE.items():
        color_mask[mask == label] = rgb

    # 1) 컬러 마스크만 저장
    Image.fromarray(color_mask).save(os.path.join(OUT_COLOR_DIR, fname))

    # 2) 원본 위에 오버레이
    img_name = os.path.splitext(fname)[0] + ".jpg"
    img_path = os.path.join(IMG_DIR, img_name)
    if not os.path.exists(img_path):

        print("[경고] 원본 이미지 없음:", img_path)
        continue

    img = np.array(Image.open(img_path).convert("RGB"))

    mask_fg = mask > 0  # H x W (2D bool)

    blended = img.astype(np.float32).copy()
    blended[mask_fg] = (
            img[mask_fg] * (1 - alpha) +
            color_mask[mask_fg] * alpha
    )

    blended = blended.astype(np.uint8)
    Image.fromarray(blended).save(os.path.join(OUT_OVERLAY_DIR, fname))

print("컬러 마스크 저장 폴더:", OUT_COLOR_DIR)
print("오버레이 저장 폴더:", OUT_OVERLAY_DIR)

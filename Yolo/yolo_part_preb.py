import os
import glob
import random
import shutil

# ===== 경로 설정 =====
PART_SAMPLE_ROOT = r"C:\Users\swu\Desktop\sample\part_sample_10000"

IMG_DIR  = os.path.join(PART_SAMPLE_ROOT, r"1.원천데이터", "part")
LBL_DIR  = os.path.join(PART_SAMPLE_ROOT, "yolo_labels_part_bbox_merged")

YOLO_ROOT  = os.path.join(PART_SAMPLE_ROOT, "yolo_part")
IMG_TRAIN  = os.path.join(YOLO_ROOT, "images", "train")
IMG_VAL    = os.path.join(YOLO_ROOT, "images", "val")
LAB_TRAIN  = os.path.join(YOLO_ROOT, "labels", "train")
LAB_VAL    = os.path.join(YOLO_ROOT, "labels", "val")

for d in [IMG_TRAIN, IMG_VAL, LAB_TRAIN, LAB_VAL]:
    os.makedirs(d, exist_ok=True)

# ===== 이미지 + 라벨 매칭 목록 만들기 =====
img_paths = glob.glob(os.path.join(IMG_DIR, "*.jpg")) + \
            glob.glob(os.path.join(IMG_DIR, "*.png"))

pairs = []
for img_path in img_paths:
    fname = os.path.basename(img_path)
    stem, _ = os.path.splitext(fname)
    lbl_path = os.path.join(LBL_DIR, stem + ".txt")

    if os.path.exists(lbl_path):
        pairs.append((img_path, lbl_path))
    else:
        print("[경고] 라벨 없음, 스킵:", fname)

print("이미지-라벨 페어 수:", len(pairs))

# ===== train / val 나누기 =====
random.seed(42)
random.shuffle(pairs)

train_ratio = 0.8
n_train = int(len(pairs) * train_ratio)

train_pairs = pairs[:n_train]
val_pairs   = pairs[n_train:]

def copy_pairs(pairs, img_dst, lbl_dst, split_name):
    cnt = 0
    for img_path, lbl_path in pairs:
        fname = os.path.basename(img_path)
        stem, _ = os.path.splitext(fname)

        shutil.copy2(img_path, os.path.join(img_dst, fname))
        shutil.copy2(lbl_path, os.path.join(lbl_dst, stem + ".txt"))
        cnt += 1

    print(f"{split_name} 세트 복사 완료: {cnt}장")

copy_pairs(train_pairs, IMG_TRAIN, LAB_TRAIN, "train")
copy_pairs(val_pairs,   IMG_VAL,   LAB_VAL,   "val")

print("최종 YOLO 데이터셋 루트:", YOLO_ROOT)

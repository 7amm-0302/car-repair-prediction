import os
import glob
import random
import shutil

TRAIN_ROOT = r"C:\Users\swu\Desktop\160._차량파손_이미지_데이터\01.데이터\1.Training"
BASE_SAMPLE_ROOT = r"C:\Users\swu\Desktop\sample"

TARGET_N_DAMAGE = 0
TARGET_N_PART   = 22000

random.seed(42)


def sample_one_set(set_name, src_img_dir, src_label_dir, target_n):
    """
    set_name   : "damage" 또는 "part"
    src_img_dir: 원천 이미지 폴더 (jpg)
    src_label_dir: 라벨 json 폴더
    target_n   : 뽑을 이미지 개수
    """

    print(f"\n=== {set_name} 세트 샘플링 시작 ===")
    print("이미지 폴더:", src_img_dir)
    print("라벨 폴더  :", src_label_dir)

    img_paths = glob.glob(os.path.join(src_img_dir, "*.jpg"))
    stems = [os.path.splitext(os.path.basename(p))[0] for p in img_paths]
    print("전체 이미지 수:", len(stems))

    if len(stems) == 0:
        print("[경고] 이미지가 0장입니다. 경로를 다시 확인하세요.")
        return

    n_sample = min(target_n, len(stems))
    sampled_stems = random.sample(stems, n_sample)
    print("샘플링된 이미지 수:", n_sample)


    dst_root = os.path.join(BASE_SAMPLE_ROOT, f"{set_name}_sample_{n_sample}")
    dst_img_dir   = os.path.join(dst_root, "1.원천데이터", set_name)
    dst_label_dir = os.path.join(dst_root, "2.라벨링데이터", set_name)

    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_label_dir, exist_ok=True)

    def safe_copy(stem, src_dir, dst_dir, ext):
        src = os.path.join(src_dir, stem + ext)
        if os.path.exists(src):
            shutil.copy2(src, dst_dir)
        else:
            print(f"[경고] 파일 없음: {src}")

    for stem in sampled_stems:
        safe_copy(stem, src_img_dir,   dst_img_dir,   ".jpg")
        safe_copy(stem, src_label_dir, dst_label_dir, ".json")

    print(f"{set_name} 세트 복사 완료! → {dst_root}")
    print("이미지:", dst_img_dir)
    print("라벨  :", dst_label_dir)


if __name__ == "__main__":
    SRC_IMG_DAMAGE = os.path.join(TRAIN_ROOT, "1.원천데이터", "TS_damage", "damage")
    SRC_LAB_DAMAGE = os.path.join(TRAIN_ROOT, "2.라벨링데이터", "damage")

    sample_one_set(
        set_name="damage",
        src_img_dir=SRC_IMG_DAMAGE,
        src_label_dir=SRC_LAB_DAMAGE,
        target_n=TARGET_N_DAMAGE,
    )

    SRC_IMG_PART = os.path.join(TRAIN_ROOT, "1.원천데이터", "TS_damage_part", "damage_part")
    SRC_LAB_PART = os.path.join(TRAIN_ROOT, "2.라벨링데이터", "damage_part")

    sample_one_set(
        set_name="part",
        src_img_dir=SRC_IMG_PART,
        src_label_dir=SRC_LAB_PART,
        target_n=TARGET_N_PART,
    )

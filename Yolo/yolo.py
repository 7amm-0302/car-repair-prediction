from ultralytics import YOLO

def main():
    model = YOLO('yolov8m.pt')

    DATA_YAML = r"C:\Users\swu\Desktop\sample\part_sample_10000\yolo_part\data.yaml"

    model.train(
        data=DATA_YAML,
        epochs=50,
        patience=20,
        imgsz=640,
        batch=8,
        name='final',
        device='0',      # GPU 0번 사용
        verbose=True
    )

    results = model.val()
    print(results)


if __name__ == '__main__':
    main()

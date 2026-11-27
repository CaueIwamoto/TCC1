from ultralytics import YOLO

model = YOLO("yolo11n.pt")

print("Iniciando treinamento...")

model.train(
    data="dataset_colorido_peq/data.yaml",
    epochs=15,
    imgsz=320,
    batch=4,
    device="cpu",
    workers=0,
    patience=5,
    project="models",
    name="Modelo 1",
    exist_ok=True,
    seed=42
)

print("Treinamento concluído!")

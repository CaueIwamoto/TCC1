from ultralytics import YOLO

model = YOLO("yolo11n.pt")

print("Iniciando treinamento...")

model.train(
    data="datasets/dataset_colorido_gnd/data.yaml",
    epochs=35,
    imgsz=320,
    batch=4,
    device="cpu",
    workers=0,
    patience=5,
    project="models",
    name="Modelo2",
    exist_ok=True,
    seed=42
)

print("Treinamento concluído!")

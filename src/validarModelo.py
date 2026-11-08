# Gerar resultados após passar pelo conjunto de teste

from ultralytics import YOLO

modelo = YOLO(
    "runs/detect/runs/overhead_person_i3/weights/best.pt"
)

resultados = modelo.val(
    data="dataset/data.yaml",
    split="test",
    imgsz=320,
    batch=2,
    device="cpu",
    workers=0
)

print("\n========== RESULTADOS ==========")

print(f"Precision: {resultados.box.mp:.4f}")
print(f"Recall: {resultados.box.mr:.4f}")
print(f"mAP50: {resultados.box.map50:.4f}")
print(f"mAP50-95: {resultados.box.map:.4f}")

print("================================")

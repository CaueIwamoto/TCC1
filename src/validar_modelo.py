
from ultralytics import YOLO

# Carrega o melhor modelo treinado
modelo = YOLO(
    "runs/detect/runs/treinamento_rapido/weights/best.pt"
)

# Avalia utilizando o conjunto de teste
resultados = modelo.val(
    data="data.yaml",
    split="test",
    imgsz=640
)

# Exibe as métricas
print("\n========== RESULTADOS ==========")

print(f"Precision: {resultados.box.mp:.4f}")
print(f"Recall: {resultados.box.mr:.4f}")
print(f"mAP50: {resultados.box.map50:.4f}")
print(f"mAP50-95: {resultados.box.map:.4f}")

print("================================")

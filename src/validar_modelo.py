
from ultralytics import YOLO

# Carrega o melhor modelo treinado
modelo = YOLO(
    "runs/detect/runs/Modelo3/weights/best.pt"
)

# Avalia utilizando o conjunto de teste
resultados = modelo.val(
    data="data.yaml",
    split="test",
    imgsz=480
)

precision = resultados.box.mp
recall = resultados.box.mr
f1_score = 2 * (precision * recall) / (precision + recall)

# Exibe as métricas
print("\n========== RESULTADOS ==========")

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1_score:.4f}")
print(f"mAP50: {resultados.box.map50:.4f}")
print(f"mAP50-95: {resultados.box.map:.4f}")

print("================================")

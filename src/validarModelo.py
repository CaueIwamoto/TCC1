
from ultralytics import YOLO

# Carregar o melhor modelo treinado
modelo = YOLO("runs/detect/models/Modelo1/weights/best.pt")

# Realizar validação no conjunto de teste
resultados = modelo.val(
    data="datasets/dataset_colorido_peq/data.yaml",
    split="test",
    imgsz=320,
    batch=4,
    device="cpu",
    workers=0
)

# Obter métricas
precision = resultados.box.mp
recall = resultados.box.mr
f1_score = 2 * (precision * recall) / (precision + recall)
map50 = resultados.box.map50
map50_95 = resultados.box.map

# Exibir resultados
print("\n========== RESULTADOS ==========")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1_score:.4f}")
print(f"mAP50:     {map50:.4f}")
print(f"mAP50-95:  {map50_95:.4f}")
print("================================")
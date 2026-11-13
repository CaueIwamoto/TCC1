
from ultralytics import YOLO

# Carregar o melhor modelo treinado
modelo = YOLO("runs/detect/runs/Modelo5/weights/best.pt")

# Realizar validação no conjunto de teste
resultados = modelo.val(
    data="dataset_person/data.yaml",
    split="test",
    imgsz=240,
    batch=4,
    device="cpu",
    workers=0
)

# Obter métricas
precision = resultados.box.mp
recall = resultados.box.mr
f1_score = 2 * (precision * recall) / (precision + recall)

# Exibir resultados
print("\n========== RESULTADOS ==========")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1_score:.4f}")
print("================================")


from ultralytics import YOLO
from pathlib import Path


# ==========================================
# CONFIGURAÇÕES PARA CPU
# ==========================================

ARQUIVO_DATASET = "data.yaml"

MODELO_INICIAL = "yolo11n.pt"

EPOCAS = 50

TAMANHO_IMAGEM = 480

BATCH_SIZE = 4



# ==========================================
# CARREGAMENTO DO MODELO
# ==========================================

print("Carregando modelo YOLO11n...")

modelo = YOLO(MODELO_INICIAL)


# ==========================================
# TREINAMENTO
# ==========================================

print("\nIniciando treinamento da IA...\n")


resultado = modelo.train(

    data=ARQUIVO_DATASET,

    epochs=EPOCAS,

    imgsz=TAMANHO_IMAGEM,

    batch=BATCH_SIZE,

    device="cpu",

    project="runs",

    name="Modelo3",

    patience=10,

    workers=0,

    plots=True,

    save=True,

    seed=42

)



# ==========================================
# RESULTADOS
# ==========================================

print("\n================================")
print("TREINAMENTO FINALIZADO!")
print("================================")

print("\nMelhor modelo salvo em:")

print(
    Path("runs/detect/runs/Modelo3/weights/best.pt")
)

print("\nPesos finais salvos em:")

print(
    Path("runs/detect/runs/Modelo3/weights/last.pt")
)

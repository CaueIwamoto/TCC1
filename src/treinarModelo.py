# Treinar modelo com base no dataset

from ultralytics import YOLO

# ==========================================
# CONFIGURAÇÕES
# ==========================================

ARQUIVO_DATASET = "dataset_kaggle_melhor/data.yaml"

MODELO_INICIAL = "yolo11n.pt"

EPOCAS = 50

TAMANHO_IMAGEM = 640

BATCH_SIZE = 4


# ==========================================
# CARREGAMENTO DO MODELO
# ==========================================

print("Carregando YOLO11n...")

modelo = YOLO(MODELO_INICIAL)


# ==========================================
# TREINAMENTO
# ==========================================

print("\nIniciando treinamento na CPU...\n")

resultado = modelo.train(

    data=ARQUIVO_DATASET,

    epochs=EPOCAS,

    imgsz=TAMANHO_IMAGEM,

    batch=BATCH_SIZE,

    device="cpu",

    workers=0,

    cache=False,

    pretrained=True,

    patience=8,

    project="runs",

    name="Modelo8",

    plots=True,

    save=True,

    close_mosaic=10,

    seed=42

)


# ==========================================
# FINALIZAÇÃO
# ==========================================

print("\n================================")
print("TREINAMENTO FINALIZADO!")
print("================================")

print(
    "Melhor modelo: "
    "runs/detect/Modelo8/weights/best.pt"
)

print(
    "Último modelo: "
    "runs/detect/Modelo8/weights/last.pt"
)

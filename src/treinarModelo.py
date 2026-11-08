# Treinar modelo com base no dataset

from ultralytics import YOLO

# ==========================================
# CONFIGURAÇÕES
# ==========================================

ARQUIVO_DATASET = "dataset/data.yaml"

MODELO_INICIAL = "yolo11n.pt"

EPOCAS = 15

TAMANHO_IMAGEM = 320

BATCH_SIZE = 2


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

    name="overhead_person_i3",

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
    "runs/detect/overhead_person_i3/weights/best.pt"
)

print(
    "Último modelo: "
    "runs/detect/overhead_person_i3/weights/last.pt"
)

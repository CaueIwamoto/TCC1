# Criar as imagens brutas pro dataset com a câmera
# Pega um vídeo já gravado e separa em várias imagens

import cv2
import os

VIDEO = "./videoDatasetBruto.mp4"
PASTA_SAIDA = "imagens_brutas"
INTERVALO_SEGUNDOS = 0.5

os.makedirs(PASTA_SAIDA, exist_ok=True)

cap = cv2.VideoCapture(VIDEO)

fps = cap.get(cv2.CAP_PROP_FPS)
intervalo_frames = max(1, int(fps * INTERVALO_SEGUNDOS))

contador = 0
imagem = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if contador % intervalo_frames == 0:
        caminho = os.path.join(
            PASTA_SAIDA,
            f"imagem_{imagem:04d}.jpg"
        )

        cv2.imwrite(caminho, frame)
        imagem += 1

    contador += 1

cap.release()

print(f"Foram extraídas {imagem} imagens.")
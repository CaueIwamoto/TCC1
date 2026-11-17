import cv2
from pathlib import Path
from datetime import datetime

PASTA = Path("dataset_real/images")
PASTA.mkdir(parents=True, exist_ok=True)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Erro ao abrir a webcam.")
    exit()

print("Pressione ESPACO para salvar uma imagem.")
print("Pressione ESC para sair.")

while True:
    ret, frame = camera.read()

    if not ret:
        print("Erro ao capturar imagem.")
        break

    cv2.putText(
        frame,
        "ESPACO: salvar imagem | ESC: sair",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("Coleta de imagens reais", frame)

    tecla = cv2.waitKey(1) & 0xFF

    if tecla == 32:  # ESPACO
        nome = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        caminho = PASTA / f"{nome}.jpg"

        cv2.imwrite(str(caminho), frame)
        print(f"Imagem salva: {caminho}")

    elif tecla == 27:  # ESC
        break

camera.release()
cv2.destroyAllWindows()
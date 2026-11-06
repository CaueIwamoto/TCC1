import cv2

# Inicializa a webcam
camera = cv2.VideoCapture(0)

# Verifica se a câmera foi aberta corretamente
if not camera.isOpened():
    print("Erro: Não foi possível acessar a webcam.")
    exit()

# Define a resolução da imagem
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:

    # Captura um quadro da webcam
    ret, frame = camera.read()

    if not ret:
        print("Erro ao capturar imagem.")
        break

    # Obtém as dimensões da imagem
    altura, largura = frame.shape[:2]

    # Define a posição da linha virtual
    linha_y = altura // 2

    # Desenha a linha virtual
    cv2.line(
        frame,
        (0, linha_y),
        (largura, linha_y),
        (0, 255, 0),
        3
    )

    # Identifica as regiões de entrada e saída
    cv2.putText(
        frame,
        "REGIAO DE ENTRADA",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    cv2.putText(
        frame,
        "REGIAO DE SAIDA",
        (20, altura - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    # Exibe a imagem
    cv2.imshow("Contador de Passageiros - TCC", frame)

    # Encerra ao pressionar ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Libera a webcam
camera.release()

# Fecha as janelas
cv2.destroyAllWindows()

import cv2
from ultralytics import YOLO

from rastreador import RastreadorCentroides
from contador import ContadorDirecional


# Carrega o modelo treinado
modelo = YOLO("runs/detect/runs/treinamento_rapido/weights/best.pt")

# Inicializa a webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Erro ao abrir a webcam.")
    exit()

CONFIANCA = 0.40

# Inicializa o rastreador e o contador
rastreador = RastreadorCentroides(
    distancia_maxima=80,
    quadros_desaparecido=10
)

contador = None

while True:
    ret, frame = camera.read()

    if not ret:
        print("Erro ao capturar imagem.")
        break

    altura, largura = frame.shape[:2]
    linha_y = altura // 2

    # Cria o contador usando a posição da linha
    if contador is None:
        contador = ContadorDirecional(linha_y)

    # Detecta as bolinhas com YOLO
    resultados = modelo.predict(
        source=frame,
        conf=CONFIANCA,
        imgsz=320,
        device="cpu",
        verbose=False
    )

    deteccoes = []

    # Extrai os centros das detecções
    for caixa in resultados[0].boxes:
        x, y, w, h = caixa.xywh[0].tolist()
        deteccoes.append((int(x), int(y)))

    # Atualiza o rastreador
    objetos = rastreador.atualizar(deteccoes)

    # Atualiza o contador de cruzamentos
    contador.atualizar(objetos)

    # Desenha a linha virtual
    cv2.line(
        frame,
        (0, linha_y),
        (largura, linha_y),
        (0, 255, 0),
        2
    )

    # Desenha os IDs e posições das bolinhas
    for objeto_id, (x, y) in objetos.items():
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        cv2.putText(
            frame,
            f"ID {objeto_id}",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    # Exibe o total de passageiros
    cv2.putText(
        frame,
        f"Passageiros: {contador.total}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Contador de Passageiros", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

camera.release()
cv2.destroyAllWindows()

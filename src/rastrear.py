
import cv2
from ultralytics import YOLO

# ==========================================
# CONFIGURAÇÕES
# ==========================================

MODEL_PATH = "models/best.pt"

CAMERA_INDEX = 0

LARGURA = 640
ALTURA = 480

# Posição da linha horizontal
LINHA_Y = 240

# ==========================================
# INICIALIZAÇÃO
# ==========================================

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(CAMERA_INDEX)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, LARGURA)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTURA)

contador = 0

# Armazena o último lado da linha de cada ID
posicoes_anteriores = {}

# IDs que já foram contabilizados
ids_contabilizados = set()

# ==========================================
# PROCESSAMENTO
# ==========================================

while True:

    sucesso, frame = cap.read()

    if not sucesso:
        break

    # Executa YOLO + ByteTrack
    resultados = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.25,
        imgsz=320,
        verbose=False
    )

    resultado = resultados[0]

    # Desenha a linha virtual
    cv2.line(
        frame,
        (0, LINHA_Y),
        (LARGURA, LINHA_Y),
        (0, 255, 255),
        2
    )

    # Verifica se existem objetos rastreados
    if resultado.boxes is not None:

        if resultado.boxes.id is not None:

            caixas = resultado.boxes.xyxy.cpu().numpy()
            ids = resultado.boxes.id.int().cpu().tolist()

            for caixa, id_objeto in zip(caixas, ids):

                x1, y1, x2, y2 = map(int, caixa)

                # Centro inferior da caixa
                centro_x = (x1 + x2) // 2
                centro_y = (y1 + y2) // 2

                # Determina o lado da linha
                if centro_y < LINHA_Y:
                    lado_atual = "acima"
                else:
                    lado_atual = "abaixo"

                # Verifica se o ID já foi visto
                if id_objeto in posicoes_anteriores:

                    lado_anterior = posicoes_anteriores[id_objeto]

                    # De cima para baixo: entrada
                    if (
                        lado_anterior == "acima"
                        and lado_atual == "abaixo"
                        and id_objeto not in ids_contabilizados
                    ):
                        contador += 1
                        ids_contabilizados.add(id_objeto)

                        print(f"ID {id_objeto}: ENTRADA")

                    # De baixo para cima: saída
                    elif (
                        lado_anterior == "abaixo"
                        and lado_atual == "acima"
                        and id_objeto not in ids_contabilizados
                    ):
                        contador -= 1
                        ids_contabilizados.add(id_objeto)

                        print(f"ID {id_objeto}: SAÍDA")

                # Atualiza a posição
                posicoes_anteriores[id_objeto] = lado_atual

                # Desenha a caixa
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Exibe o ID
                cv2.putText(
                    frame,
                    f"ID: {id_objeto}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                # Desenha o centro
                cv2.circle(
                    frame,
                    (centro_x, centro_y),
                    4,
                    (0, 0, 255),
                    -1
                )

    # ======================================
    # EXIBIÇÃO DO CONTADOR
    # ======================================

    cv2.rectangle(
        frame,
        (10, 10),
        (250, 65),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"Passageiros: {contador}",
        (20, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow("Prototipo - YOLO + ByteTrack", frame)

    # ESC encerra o programa
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ==========================================
# FINALIZAÇÃO
# ==========================================

cap.release()
cv2.destroyAllWindows()

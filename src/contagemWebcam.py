# Faz a detecção de pessoas

import cv2
from ultralytics import YOLO

# ==========================================
# CONFIGURACOES
# ==========================================

MODEL_PATH = "runs/detect/runs/Modelo3/weights/best.pt"

IMG_SIZE = 320
CONFIDENCE = 0.25

# Distancia de seguranca ao redor da linha
MARGIN = 10

# ==========================================
# CARREGAR MODELO
# ==========================================

modelo = YOLO(MODEL_PATH)

# Contadores
entradas = 0
saidas = 0
ocupacao = 0

# Guarda o ultimo lado em que cada pessoa foi vista
ultimo_lado = {}

# Guarda os IDs que ja foram contabilizados
ids_contabilizados = set()

# ==========================================
# INICIAR WEBCAM E RASTREAMENTO
# ==========================================

# source é qual câmera usar. 0 = webcam externa, 1 = webcam do computador
resultados = modelo.track(
    source=0,
    stream=True,
    persist=True,
    tracker="bytetrack.yaml",
    imgsz=IMG_SIZE,
    conf=CONFIDENCE,
    device="cpu",
    verbose=False
)

# ==========================================
# PROCESSAR FRAMES
# ==========================================

for resultado in resultados:

    frame = resultado.orig_img.copy()

    altura, largura = frame.shape[:2]

    # Linha horizontal no meio da imagem
    linha_y = altura // 2

    # Desenhar linha virtual
    cv2.line(
        frame,
        (0, linha_y),
        (largura, linha_y),
        (0, 255, 0),
        2
    )

    # Obter caixas e IDs
    caixas = resultado.boxes

    if caixas is not None and caixas.id is not None:

        coordenadas = caixas.xyxy.cpu().numpy()
        ids = caixas.id.int().cpu().tolist()

        for caixa, pessoa_id in zip(coordenadas, ids):

            x1, y1, x2, y2 = map(int, caixa)

            # Centro da pessoa
            centro_x = (x1 + x2) // 2
            centro_y = (y1 + y2) // 2

            # Desenhar caixa delimitadora
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            # Desenhar centro
            cv2.circle(
                frame,
                (centro_x, centro_y),
                4,
                (0, 0, 255),
                -1
            )

            # Mostrar ID
            cv2.putText(
                frame,
                f"ID: {pessoa_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

            # ==================================
            # VERIFICAR LADO DA LINHA
            # ==================================

            if centro_y < linha_y - MARGIN:
                lado_atual = -1

            elif centro_y > linha_y + MARGIN:
                lado_atual = 1

            else:
                # Ignorar a regiao muito proxima da linha
                continue

            # ==================================
            # VERIFICAR CRUZAMENTO
            # ==================================

            if pessoa_id in ultimo_lado:

                lado_anterior = ultimo_lado[pessoa_id]

                # So contabilizar uma vez por ID
                if (
                    lado_anterior != lado_atual
                    and pessoa_id not in ids_contabilizados
                ):

                    # De cima para baixo: entrada
                    if lado_anterior == -1 and lado_atual == 1:

                        entradas += 1
                        ocupacao += 1

                        ids_contabilizados.add(pessoa_id)

                        print(f"Pessoa {pessoa_id} entrou!")

                    # De baixo para cima: saida
                    elif lado_anterior == 1 and lado_atual == -1:

                        saidas += 1
                        ocupacao -= 1

                        ids_contabilizados.add(pessoa_id)

                        print(f"Pessoa {pessoa_id} saiu!")

            # Atualizar ultimo lado conhecido
            ultimo_lado[pessoa_id] = lado_atual

    # ==========================================
    # EXIBIR CONTADORES
    # ==========================================

    cv2.putText(
        frame,
        f"Entradas: {entradas}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Saidas: {saidas}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        f"Ocupacao: {ocupacao}",
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Exibir imagem
    cv2.imshow("Contador de Passageiros", frame)

    # ESC encerra
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()

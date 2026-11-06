import cv2
import numpy as np
from ultralytics import YOLO

# ==========================================
# CONFIGURAÇÕES
# ==========================================

MODEL_PATH = "runs/detect/train/weights/best.pt"

CONFIDENCE = 0.35
MAX_DISTANCE = 70
MAX_DISAPPEARED = 10

# Contadores
entradas = 0
saidas = 0

# ==========================================
# CARREGAR MODELO
# ==========================================

model = YOLO(MODEL_PATH)

# ==========================================
# RASTREADOR DE OBJETOS
# ==========================================

class CentroidTracker:

    def __init__(self):
        self.objects = {}
        self.next_id = 0

    def register(self, centroid, bbox):
        self.objects[self.next_id] = {
            "centroid": centroid,
            "bbox": bbox,
            "missed": 0
        }

        self.next_id += 1

    def update(self, detections):

        # Caso não existam detecções
        if len(detections) == 0:

            for obj_id in list(self.objects.keys()):
                self.objects[obj_id]["missed"] += 1

                if self.objects[obj_id]["missed"] > MAX_DISAPPEARED:
                    del self.objects[obj_id]

            return self.objects

        # Caso seja a primeira detecção
        if len(self.objects) == 0:

            for det in detections:
                self.register(det[0], det[1])

            return self.objects

        # Calcular distâncias entre objetos existentes
        # e novas detecções

        pairs = []

        for obj_id, obj in self.objects.items():

            old_x, old_y = obj["centroid"]

            for index, det in enumerate(detections):

                new_x, new_y = det[0]

                distance = np.sqrt(
                    (old_x - new_x) ** 2 +
                    (old_y - new_y) ** 2
                )

                if distance <= MAX_DISTANCE:
                    pairs.append((distance, obj_id, index))

        # Associar as menores distâncias primeiro
        pairs.sort(key=lambda x: x[0])

        used_objects = set()
        used_detections = set()

        for distance, obj_id, index in pairs:

            if obj_id in used_objects:
                continue

            if index in used_detections:
                continue

            old_y = self.objects[obj_id]["centroid"][1]

            new_centroid = detections[index][0]
            new_bbox = detections[index][1]

            # Atualizar posição
            self.objects[obj_id]["centroid"] = new_centroid
            self.objects[obj_id]["bbox"] = new_bbox
            self.objects[obj_id]["missed"] = 0

            # Verificar cruzamento da linha virtual
            check_crossing(obj_id, old_y, new_centroid[1])

            used_objects.add(obj_id)
            used_detections.add(index)

        # Registrar novas detecções
        for index, det in enumerate(detections):

            if index not in used_detections:
                self.register(det[0], det[1])

        # Remover objetos desaparecidos
        for obj_id in list(self.objects.keys()):

            if obj_id not in used_objects and obj_id not in [
                obj_id2 for _, obj_id2, _ in pairs
            ]:
                self.objects[obj_id]["missed"] += 1

            if self.objects[obj_id]["missed"] > MAX_DISAPPEARED:
                del self.objects[obj_id]

        return self.objects


# ==========================================
# CONTROLE DE CRUZAMENTO
# ==========================================

LINE_Y = 240

counted_objects = set()


def check_crossing(obj_id, old_y, new_y):

    global entradas, saidas

    if obj_id in counted_objects:
        return

    # Movimento de cima para baixo
    if old_y < LINE_Y and new_y >= LINE_Y:

        entradas += 1
        counted_objects.add(obj_id)

        print(f"Entrada! Total: {entradas}")

    # Movimento de baixo para cima
    elif old_y > LINE_Y and new_y <= LINE_Y:

        saidas += 1
        counted_objects.add(obj_id)

        print(f"Saída! Total: {saidas}")


# ==========================================
# INICIALIZAR CÂMERA
# ==========================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

tracker = CentroidTracker()

# ==========================================
# LOOP PRINCIPAL
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Erro ao acessar a câmera.")
        break

    height, width = frame.shape[:2]

    # Ajustar posição da linha ao tamanho da imagem
    LINE_Y = height // 2

    # ======================================
    # DETECÇÃO YOLO
    # ======================================

    results = model.predict(
        frame,
        conf=CONFIDENCE,
        imgsz=640,
        verbose=False
    )

    detections = []

    for box in results[0].boxes:

        x1, y1, x2, y2 = map(
            int, box.xyxy[0].tolist()
        )

        confidence = float(box.conf[0])

        # Centro da bolinha
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        detections.append((
            (cx, cy),
            (x1, y1, x2, y2)
        ))

    # ======================================
    # RASTREAMENTO
    # ======================================

    objects = tracker.update(detections)

    # ======================================
    # DESENHAR LINHA VIRTUAL
    # ======================================

    cv2.line(
        frame,
        (0, LINE_Y),
        (width, LINE_Y),
        (0, 255, 255),
        2
    )

    # ======================================
    # DESENHAR OBJETOS
    # ======================================

    for obj_id, obj in objects.items():

        cx, cy = obj["centroid"]

        x1, y1, x2, y2 = obj["bbox"]

        # Caixa de detecção
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Centro do objeto
        cv2.circle(
            frame,
            (cx, cy),
            4,
            (0, 0, 255),
            -1
        )

        # Identificador
        cv2.putText(
            frame,
            f"ID: {obj_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )

    # ======================================
    # EXIBIR CONTADORES
    # ======================================

    cv2.rectangle(
        frame,
        (10, 10),
        (240, 100),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"Entradas: {entradas}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Saidas: {saidas}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    cv2.imshow(
        "Contagem de Passageiros",
        frame
    )

    # ESC para encerrar
    if cv2.waitKey(1) & 0xFF == 27:
        break


# ==========================================
# FINALIZAR
# ==========================================

cap.release()
cv2.destroyAllWindows()

print("\nRESULTADO FINAL")
print("Entradas:", entradas)
print("Saidas:", saidas)
print("Ocupacao:", entradas - saidas)
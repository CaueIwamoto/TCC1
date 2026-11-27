
import cv2
import numpy as np
import random
from pathlib import Path

# ==========================================
# CONFIGURAÇÕES
# ==========================================

LARGURA = 640
ALTURA = 480

TOTAL_IMAGENS = 500

DIRETORIO_DATASET = Path("dataset_colorido_peq")

DIVISAO = {
    "train": 350,
    "val": 75,
    "test": 75
}


# ==========================================
# CRIAÇÃO DAS PASTAS
# ==========================================

for conjunto in DIVISAO:

    pasta_imagens = DIRETORIO_DATASET / "images" / conjunto
    pasta_labels = DIRETORIO_DATASET / "labels" / conjunto

    pasta_imagens.mkdir(parents=True, exist_ok=True)
    pasta_labels.mkdir(parents=True, exist_ok=True)


# ==========================================
# GERAÇÃO DE UM FUNDO COLORIDO
# ==========================================

def gerar_fundo():

    # Escolhe uma cor aleatória em HSV
    tonalidade = random.randint(0, 179)

    # Varia a saturação e o brilho
    saturacao = random.randint(40, 180)
    brilho = random.randint(70, 240)

    cor = np.uint8([[[tonalidade, saturacao, brilho]]])

    # Converte HSV para BGR
    cor_bgr = cv2.cvtColor(
        cor,
        cv2.COLOR_HSV2BGR
    )[0][0]

    # Cria pequenas variações de iluminação
    gradiente = np.linspace(
        random.randint(-15, 0),
        random.randint(0, 15),
        LARGURA
    )

    fundo = np.zeros(
        (ALTURA, LARGURA, 3),
        dtype=np.float32
    )

    for canal in range(3):

        fundo[:, :, canal] = (
            cor_bgr[canal] + gradiente
        )

    imagem = np.clip(
        fundo, 0, 255
    ).astype(np.uint8)

    return imagem


# ==========================================
# GERAÇÃO DE UMA IMAGEM
# ==========================================

def gerar_imagem():

    # Fundo com cor aleatória
    imagem = gerar_fundo()

    # Quantidade de bolinhas
    quantidade = random.randint(0, 8)

    # Algumas imagens serão vazias
    if random.random() < 0.05:
        quantidade = 0

    bolinhas = []
    anotacoes = []

    for _ in range(quantidade):

        raio = random.randint(9, 19)

        x = random.randint(
            raio + 2,
            LARGURA - raio - 3
        )

        y = random.randint(
            raio + 2,
            ALTURA - raio - 3
        )

        # Evita sobreposição excessiva
        sobreposicao = False

        for bx, by, br in bolinhas:

            distancia = np.sqrt(
                (x - bx) ** 2 +
                (y - by) ** 2
            )

            if distancia < raio + br + 5:
                sobreposicao = True
                break

        if sobreposicao:
            continue

        bolinhas.append((x, y, raio))

        # Desenha a bolinha preta
        intensidade = random.randint(0, 20)

        cv2.circle(
            imagem,
            (x, y),
            raio,
            (intensidade,) * 3,
            -1,
            cv2.LINE_AA
        )

        # Coordenadas normalizadas para YOLO
        centro_x = x / LARGURA
        centro_y = y / ALTURA

        largura = (2 * raio) / LARGURA
        altura = (2 * raio) / ALTURA

        anotacoes.append(
            f"0 {centro_x:.6f} {centro_y:.6f} "
            f"{largura:.6f} {altura:.6f}"
        )

    # Adiciona ruído
    ruido = np.random.normal(
        0,
        random.uniform(0, 2.5),
        imagem.shape
    ).astype(np.float32)

    imagem = np.clip(
        imagem.astype(np.float32) + ruido,
        0,
        255
    ).astype(np.uint8)

    # Pequenas variações de foco
    if random.random() < 0.2:

        imagem = cv2.GaussianBlur(
            imagem,
            (3, 3),
            0
        )

    return imagem, anotacoes


# ==========================================
# GERAÇÃO DO DATASET
# ==========================================

numero_imagem = 0

for conjunto, quantidade in DIVISAO.items():

    pasta_imagens = (
        DIRETORIO_DATASET / "images" / conjunto
    )

    pasta_labels = (
        DIRETORIO_DATASET / "labels" / conjunto
    )

    print(f"\nGerando conjunto: {conjunto}")

    for i in range(quantidade):

        numero_imagem += 1

        imagem, anotacoes = gerar_imagem()

        nome = f"imagem_{i:04d}"

        # Salva a imagem
        cv2.imwrite(
            str(pasta_imagens / f"{nome}.jpg"),
            imagem
        )

        # Salva as anotações
        with open(
            pasta_labels / f"{nome}.txt",
            "w"
        ) as arquivo:

            arquivo.write("\n".join(anotacoes))

    print(f"{quantidade} imagens geradas!")

print("\n================================")
print("DATASET GERADO COM SUCESSO!")
print(f"Total de imagens: {numero_imagem}")
print("================================")

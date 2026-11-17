
import cv2
import numpy as np
import random
from pathlib import Path

# ==========================================
# CONFIGURAÇÕES
# ==========================================

LARGURA = 640
ALTURA = 480

TOTAL_IMAGENS = 300

DIRETORIO_DATASET = Path("dataset")

DIVISAO = {
    "train": 210,
    "val": 60,
    "test": 30
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
# GERAÇÃO DE UMA IMAGEM
# ==========================================

def gerar_imagem():

    # Cria um fundo branco com pequenas
    # variações de iluminação

    base = random.randint(235, 255)

    gradiente = np.linspace(
        random.randint(-5, 0),
        random.randint(0, 5),
        LARGURA
    )

    fundo = np.tile(
        np.clip(base + gradiente, 0, 255).astype(np.uint8),
        (ALTURA, 1)
    )

    imagem = cv2.cvtColor(
        fundo,
        cv2.COLOR_GRAY2BGR
    )

    # Quantidade de bolinhas na imagem
    quantidade = random.randint(0, 8)

    # Algumas imagens serão vazias
    # para ensinar o modelo a não detectar objetos inexistentes
    if random.random() < 0.05:
        quantidade = 0

    bolinhas = []
    anotacoes = []

    for _ in range(quantidade):

        raio = random.randint(9, 19)

        # Evita que as bolinhas fiquem cortadas
        x = random.randint(raio + 2, LARGURA - raio - 3)
        y = random.randint(raio + 2, ALTURA - raio - 3)

        # Evita sobreposição excessiva
        sobreposicao = False

        for bx, by, br in bolinhas:

            distancia = np.sqrt(
                (x - bx) ** 2 + (y - by) ** 2
            )

            if distancia < raio + br + 5:
                sobreposicao = True
                break

        if sobreposicao:
            continue

        bolinhas.append((x, y, raio))

        # Desenha a bolinha preta
        cv2.circle(
            imagem,
            (x, y),
            raio,
            (random.randint(0, 15),) * 3,
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

    # Adiciona um pequeno ruído na imagem
    ruido = np.random.normal(
        0, random.uniform(0, 2.5),
        imagem.shape
    ).astype(np.float32)

    imagem = np.clip(
        imagem.astype(np.float32) + ruido,
        0, 255
    ).astype(np.uint8)

    # Pequenas variações de foco
    if random.random() < 0.2:
        imagem = cv2.GaussianBlur(
            imagem, (3, 3), 0
        )

    return imagem, anotacoes


# ==========================================
# GERAÇÃO DO DATASET
# ==========================================

numero_imagem = 0

for conjunto, quantidade in DIVISAO.items():

    pasta_imagens = DIRETORIO_DATASET / "images" / conjunto
    pasta_labels = DIRETORIO_DATASET / "labels" / conjunto

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

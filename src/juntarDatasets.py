
import shutil
from pathlib import Path

# Pastas de origem
DATASET_PROPRIO = Path("datasets/dataset_person_split")
DATASET_PUBLICO = Path("datasets/dataset_kaggle_split")

# Pasta de destino
DATASET_FINAL = Path("datasets/dataset_person_kaggle")

# Divisões
DIVISOES = ["train", "valid", "test"]

# Extensões de imagens aceitas
EXTENSOES = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def copiar_dataset(origem, destino, prefixo):

    total = 0

    for divisao in DIVISOES:

        pasta_imagens = origem / divisao / "images"
        pasta_labels = origem / divisao / "labels"

        destino_imagens = destino / divisao / "images"
        destino_labels = destino / divisao / "labels"

        destino_imagens.mkdir(parents=True, exist_ok=True)
        destino_labels.mkdir(parents=True, exist_ok=True)

        if not pasta_imagens.exists():
            raise FileNotFoundError(pasta_imagens)

        if not pasta_labels.exists():
            raise FileNotFoundError(pasta_labels)

        for imagem in pasta_imagens.iterdir():

            if imagem.suffix.lower() not in EXTENSOES:
                continue

            anotacao = pasta_labels / (imagem.stem + ".txt")

            if not anotacao.exists():
                raise FileNotFoundError(
                    f"Anotacao ausente: {anotacao}"
                )

            # Confere os identificadores das classes
            with open(anotacao, "r", encoding="utf-8") as arquivo:
                for numero_linha, linha in enumerate(arquivo, 1):

                    if not linha.strip():
                        continue

                    classe = int(linha.split()[0])

                    if classe != 0:
                        raise ValueError(
                            f"Classe diferente de 0 em {anotacao}, "
                            f"linha {numero_linha}"
                        )

            # Novo nome para evitar arquivos repetidos
            novo_nome = prefixo + "_" + imagem.name

            novo_label = prefixo + "_" + anotacao.name

            shutil.copy2(
                imagem,
                destino_imagens / novo_nome
            )

            shutil.copy2(
                anotacao,
                destino_labels / novo_label
            )

            total += 1

    return total


# Combinar datasets
quantidade_propria = copiar_dataset(
    DATASET_PROPRIO,
    DATASET_FINAL,
    "proprio"
)

quantidade_publica = copiar_dataset(
    DATASET_PUBLICO,
    DATASET_FINAL,
    "publico"
)

print("\n========== DATASETS COMBINADOS ==========")
print(f"Imagens próprias: {quantidade_propria}")
print(f"Imagens públicas: {quantidade_publica}")
print(f"Total: {quantidade_propria + quantidade_publica}")
print("=========================================")

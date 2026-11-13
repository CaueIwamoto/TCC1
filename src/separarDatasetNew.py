# Divide o dataset em train, test e split considerando ordem
# para não criar problemas no futuro

import os
import shutil
from pathlib import Path

# Pastas originais
PASTA_IMAGENS = Path("dataset/images")
PASTA_LABELS = Path("dataset/labels")

# Pasta de destino
PASTA_DESTINO = Path("dataset_person")

# Proporções
PROPORCAO_TREINO = 0.70
PROPORCAO_VALIDACAO = 0.15
PROPORCAO_TESTE = 0.15

# Extensões aceitas
EXTENSOES = [".jpg", ".jpeg", ".png"]

# Localizar imagens
imagens = sorted(
    [p for p in PASTA_IMAGENS.iterdir()
     if p.suffix.lower() in EXTENSOES],
    key=lambda p: int(p.stem.split("_")[-1])
)

total = len(imagens)

if total == 0:
    raise ValueError("Nenhuma imagem encontrada.")

# Calcular quantidades
qtd_treino = int(total * PROPORCAO_TREINO)
qtd_validacao = int(total * PROPORCAO_VALIDACAO)

# Separar em blocos temporais
conjuntos = {
    "train": imagens[:qtd_treino],
    "valid": imagens[qtd_treino:qtd_treino + qtd_validacao],
    "test": imagens[qtd_treino + qtd_validacao:]
}

# Criar estrutura e copiar arquivos
for conjunto, lista_imagens in conjuntos.items():

    pasta_img = PASTA_DESTINO / conjunto / "images"
    pasta_lbl = PASTA_DESTINO / conjunto / "labels"

    pasta_img.mkdir(parents=True, exist_ok=True)
    pasta_lbl.mkdir(parents=True, exist_ok=True)

    for imagem in lista_imagens:

        arquivo_label = PASTA_LABELS / (imagem.stem + ".txt")

        if not arquivo_label.exists():
            raise FileNotFoundError(
                f"Anotação não encontrada: {arquivo_label}"
            )

        shutil.copy2(imagem, pasta_img / imagem.name)
        shutil.copy2(arquivo_label, pasta_lbl / arquivo_label.name)

    print(f"{conjunto}: {len(lista_imagens)} imagens")

print("\nDivisão concluída!")
print(f"Total: {total} imagens")

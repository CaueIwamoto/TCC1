# Transformar uma pasta de dataset bagunçada em uma separa por treino, validação e teste
# OBS: Depois de separar, garantir que data.yaml está na mesma pasta do diretório

import os
import random
import shutil
from pathlib import Path

# Caminho do dataset original
dataset_dir = Path("dataset zoado")

images_dir = dataset_dir / "images"
labels_dir = dataset_dir / "labels"

# Caminho do novo dataset
output_dir = Path("dataset")

# Proporções
train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

# Seed para reproduzir a divisão
random.seed(42)

# Extensões de imagens aceitas
extensions = [".jpg", ".jpeg", ".png", ".bmp"]

# Lista de imagens
images = [
    f for f in images_dir.iterdir()
    if f.suffix.lower() in extensions
]

# Embaralhar
random.shuffle(images)

# Quantidades
total = len(images)

train_count = int(total * train_ratio)
val_count = int(total * val_ratio)

# Separar
train_images = images[:train_count]
val_images = images[train_count:train_count + val_count]
test_images = images[train_count + val_count:]

# Função para copiar imagens e labels
def copy_split(files, split):

    img_output = output_dir / "images" / split
    lbl_output = output_dir / "labels" / split

    img_output.mkdir(parents=True, exist_ok=True)
    lbl_output.mkdir(parents=True, exist_ok=True)

    for img in files:

        # Copiar imagem
        shutil.copy2(img, img_output / img.name)

        # Encontrar anotação correspondente
        label = labels_dir / (img.stem + ".txt")

        if label.exists():
            shutil.copy2(label, lbl_output / label.name)
        else:
            print(f"Anotação não encontrada: {img.name}")

# Executar divisão
copy_split(train_images, "train")
copy_split(val_images, "val")
copy_split(test_images, "test")

# Resultados
print("\nDivisão concluída!")
print(f"Total: {total}")
print(f"Train: {len(train_images)}")
print(f"Val: {len(val_images)}")
print(f"Test: {len(test_images)}")

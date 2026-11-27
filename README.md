# Modelo 1

| Parâmetro         | Valor utilizado      |
|-------------------|----------------------|
| Dataset           | dataset_colorido_peq |
| Modelo            | YOLO11n              |
| Épocas            | 15                   |
| Tamanho da Imagem | 320                  |
| Batch Size        | 4                    |
| Paciência         | 5                    |

| Resultado | Valor  |
|-----------|--------|
| Precisão  | 0.9996 |
| Recall    | 1.0000 |
| F1-score  | 0.9998 |
| mAP50     | 1.0000 |
| mAP50-95  | 0.9099 |

Observações:
- Esse modelo foi só para testar o funcionamento do ByteTrack, por isso é simples
- Mesmo com pouco treinamento, teve um resultado excelente, e rastreia bem as bolinhas mesmo quando passam com certa velocidade

<br>

# Datasets
## dataset_colorido_peq:
- Dataset sintético de bolinhas pretas com fundo variando a cor
- As bolinhas pretas podem aparecer de 0 a 8 vezes em cada imagem, variando posição e tamanho

| Parâmetro | Valor utilizado |
|-----------|-----------------|
| Treino    | 350             |
| Validação | 75              |
| Teste     | 75              |
| Divisão   | 70-15-15        |
| Total     | 500             |
| Largura   | 640             |
| Altura    | 480             |

<br>


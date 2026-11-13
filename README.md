# TCC1
Trabalho Final de Curso I

Link da apresentação:
https://1drv.ms/p/c/dff90971fbc98f17/IQDI3fkWryXrTbeC8F3JWxUEAUDkAzZuyNZeGDHjNa8TE-o?e=tapB8D

Link da conversa scispace (Sérgio):
https://scispace.com/share/43ef0e08-e704-437d-b2fa-06ed7a89a432

Link da conversa scispace (Hanna): 
https://scispace.com/share/70521f6a-edf4-486b-9e26-b85162f08b52

## Modelos
### Modelo 1

| Parâmetro | Valor utilizado |
| ------------- | ------------- |
| Dataset  | Overhead Person Model  |
| Modelo  | YOLO11n  |
| Épocas | 15 |
| Tamanho da Imagem | 320 |
| Batch Size | 2 |
| Paciência | 8 |

| Resultado | Valor |
|-|-|
| Precisão | 0.9566 |
| Recall | 0.9573 |
| mAP50 | 0.9694 |
| mAP50-95 | 0.5637 |

Observações:
- Apesar de uma tentativa falha da Microsoft/Bill Gates de sabotar meu TCC, consegui jogar no GitHub e derrotar essa corporação maldita
- O modelo não reconheceu pessoas direito, apenas o Pereba. Às vezes reconhecia uma cabeça, mas não era muito consistente. Acho que o problema é mais o dataset do que qualquer outra coisa, que pega as imagens muito de cima. A câmera estava a ~2.60m de distância do chão, sendo que o ideal é ~2.10m, e mesmo com a distância maior, não teve um resultado muito bom
- Chupa Bill Gates

<br>

### Modelo 2

| Parâmetro | Valor utilizado |
| ------------- | ------------- |
| Dataset  | Overhead Person Model  |
| Modelo  | YOLO11n  |
| Épocas | 30 |
| Tamanho da Imagem | 480 |
| Batch Size | 2 |
| Paciência | 8 |

| Resultado | Valor |
|-|-|
| Precisão | 0.9639 |
| Recall | 0.9612 |
| mAP50 | 0.9759 |
| mAP50-95 | 0.6166 |

Observações:
- Dobrei o número de épocas e aumentei a resolução e mesmo assim não reconhece uma pessoa. Isso indica que o problema realmente é o dataset, e não os parâmetros

<br>

### Modelo 3

| Parâmetro | Valor utilizado |
| ------------- | ------------- |
| Dataset  | Dataset Person (Dataset próprio) |
| Modelo  | YOLO11n  |
| Épocas | 10 |
| Tamanho da Imagem | 320 |
| Batch Size | 2 |
| Paciência | 8 |

| Resultado | Valor |
|-|-|
| Precisão | 0.8262 |
| Recall | 0.5882 |
| F1-score | 0.6872 |

Observações:
- Esse foi nosso primeiro modelo que usa um dataset próprio. Fiz com poucas épocas, batch, etc. porque só queria testar se era eficaz
- Os resultados não foram muito bons. Toda hora, o modelo detectava várias pessoas em um mesmo lugar, quando realmente havia apenas uma. Mesmo assim, foi bom para ver que um dataset próprio provavelmente é a solução

<br>

### Modelo 4

| Parâmetro | Valor utilizado |
| ------------- | ------------- |
| Dataset  | Dataset Person (Dataset próprio) |
| Modelo  | YOLO11n  |
| Épocas | 30 |
| Tamanho da Imagem | 480 |
| Batch Size | 4 |
| Paciência | 8 |

| Resultado | Valor |
|-|-|
| Precisão | 0.7745 |
| Recall | 0.6471 |
| F1-score | 0.7051 |

<br>

### Modelo 5

| Parâmetro | Valor utilizado |
| ------------- | ------------- |
| Dataset  | Dataset Person (Dataset próprio) |
| Modelo  | YOLO11n  |
| Épocas | 30 |
| Tamanho da Imagem | 240 |
| Batch Size | 4 |
| Paciência | 8 |

| Resultado | Valor |
|-|-|
| Precisão | 0.7582 |
| Recall | 0.7647 |
| F1-score | 0.7614 |

Observações (Modelos 4 e 5):
- Nenhum dos dois modelos melhorou muito comparados com o 3
- Esses dois modelos foram para comparar a diferença que a resolução faz, mas acho que não teve muita, pelo menos com os valores que eu usei
- Acho que o problema não é o dataset, e sim os parâmetros

<br>

### Modelo 6

| Parâmetro | Valor utilizado |
| ------------- | ------------- |
| Dataset  | Dataset Person (Dataset próprio) |
| Modelo  | YOLO11n  |
| Épocas | 30 |
| Tamanho da Imagem | 640 |
| Batch Size | 4 |
| Paciência | 8 |

| Resultado | Valor |
|-|-|
| Precisão | 0.7630 |
| Recall | 0.7578 |
| F1-score | 0.7604 |

Observações:
- Continua igual, acho que chegou a hora de achar um dataset bom ou fazer um melhor...

## Datasets maneiros
[Top View People](https://universe.roboflow.com/top-view-pedestrians/top-view-people)

[Human Detection Top View](https://universe.roboflow.com/paras-rxcr4/human-detection-top-view)

[Overhead Person Model](https://universe.roboflow.com/riccardo-kxtut/overhead-person-szky0)

## Site de anotação
[annotation.photo](https://annotate.photo/)

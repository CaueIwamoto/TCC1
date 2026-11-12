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

## Datasets maneiros
[Top View People](https://universe.roboflow.com/top-view-pedestrians/top-view-people)

[Human Detection Top View](https://universe.roboflow.com/paras-rxcr4/human-detection-top-view)

[Overhead Person Model](https://universe.roboflow.com/riccardo-kxtut/overhead-person-szky0)

## Site de anotação
[annotation.photo](https://annotate.photo/)

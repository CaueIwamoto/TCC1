
import math


class RastreadorCentroides:
    def __init__(self, distancia_maxima=80, quadros_desaparecido=10):
        self.proximo_id = 0
        self.centroides = {}
        self.desaparecidos = {}

        self.distancia_maxima = distancia_maxima
        self.quadros_desaparecido = quadros_desaparecido

    def registrar(self, centro):
        objeto_id = self.proximo_id
        self.centroides[objeto_id] = centro
        self.desaparecidos[objeto_id] = 0
        self.proximo_id += 1
        return objeto_id

    def remover(self, objeto_id):
        del self.centroides[objeto_id]
        del self.desaparecidos[objeto_id]

    def atualizar(self, deteccoes):
        # Sem detecções: mantém os objetos por alguns quadros
        if len(deteccoes) == 0:
            for objeto_id in list(self.desaparecidos.keys()):
                self.desaparecidos[objeto_id] += 1

                if self.desaparecidos[objeto_id] > self.quadros_desaparecido:
                    self.remover(objeto_id)

            return {}

        # Primeiro quadro: registra todas as bolinhas
        if len(self.centroides) == 0:
            visiveis = {}

            for centro in deteccoes:
                objeto_id = self.registrar(centro)
                visiveis[objeto_id] = centro

            return visiveis

        # Calcula as distâncias entre objetos conhecidos e novas detecções
        distancias = []

        for objeto_id, centro_antigo in self.centroides.items():
            for indice, centro_novo in enumerate(deteccoes):
                distancia = math.dist(centro_antigo, centro_novo)
                distancias.append((distancia, objeto_id, indice))

        # Prioriza as associações com menor distância
        distancias.sort(key=lambda item: item[0])

        ids_associados = set()
        deteccoes_associadas = set()
        visiveis = {}

        for distancia, objeto_id, indice in distancias:
            if distancia > self.distancia_maxima:
                break

            if objeto_id in ids_associados:
                continue

            if indice in deteccoes_associadas:
                continue

            centro = deteccoes[indice]

            self.centroides[objeto_id] = centro
            self.desaparecidos[objeto_id] = 0

            ids_associados.add(objeto_id)
            deteccoes_associadas.add(indice)
            visiveis[objeto_id] = centro

        # Atualiza o tempo de desaparecimento dos objetos não associados
        for objeto_id in list(self.centroides.keys()):
            if objeto_id not in ids_associados:
                self.desaparecidos[objeto_id] += 1

                if self.desaparecidos[objeto_id] > self.quadros_desaparecido:
                    self.remover(objeto_id)

        # Registra novas bolinhas que não foram associadas
        for indice, centro in enumerate(deteccoes):
            if indice not in deteccoes_associadas:
                objeto_id = self.registrar(centro)
                visiveis[objeto_id] = centro

        return visiveis

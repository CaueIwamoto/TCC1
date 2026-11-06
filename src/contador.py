
class ContadorDirecional:
    def __init__(self, linha_y, margem=10):
        self.linha_y = linha_y
        self.margem = margem

        self.total = 0
        self.ultimo_lado = {}

    def atualizar(self, objetos):
        for objeto_id, (x, y) in objetos.items():

            # Identifica o lado da linha em que a bolinha está
            if y < self.linha_y - self.margem:
                lado_atual = -1

            elif y > self.linha_y + self.margem:
                lado_atual = 1

            else:
                # Ignora a região imediatamente próxima à linha
                continue

            lado_anterior = self.ultimo_lado.get(objeto_id)

            # De cima para baixo: entrada
            if lado_anterior == -1 and lado_atual == 1:
                self.total += 1
                print(f"ID {objeto_id}: entrada (+1)")

            # De baixo para cima: saída
            elif lado_anterior == 1 and lado_atual == -1:
                self.total -= 1
                print(f"ID {objeto_id}: saída (-1)")

            self.ultimo_lado[objeto_id] = lado_atual

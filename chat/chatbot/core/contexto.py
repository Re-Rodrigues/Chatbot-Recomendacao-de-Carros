class Contexto:
    def __init__(self):
        self.ultima_intencao = None
        self.carros = []
        self.carro_foco = None
        self.previous_carros = []
        self.marca_foco = None
        self.carros_marca_pool = []

    def reset(self, intencao):
        self.ultima_intencao = intencao
        self.carro_foco = None
        self.carros = []
        self.previous_carros = []

    def reset_marca(self, marca):
        self.marca_foco = marca
        self.carros_marca_pool = []

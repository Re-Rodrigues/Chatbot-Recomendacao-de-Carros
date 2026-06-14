class Contexto:
    def __init__(self):
        self.ultima_intencao = None
        self.carros = []
        self.carro_foco = None
        self.previous_carros = []
        self.marca_foco = None
        self.carros_marca_pool = []
        # Estado para buscas paginadas no CarFit
        self.carfit_offset = 0
        self.carfit_previous_ids = []
        self.carfit_query = None

    def reset(self, intencao):
        self.ultima_intencao = intencao
        self.carro_foco = None
        self.carros = []
        self.previous_carros = []
        self.carfit_offset = 0
        self.carfit_previous_ids = []
        self.carfit_query = None

    def reset_marca(self, marca):
        self.marca_foco = marca
        self.carros_marca_pool = []

class Tracer:
    def __init__(self):
        self.calls = []

    def add(self, name: str):
        self.calls.append(name)
class Vertex():
    def __init__(self, value):
        self.value = value
        self.edges = {}
    def get_edges(self):
        return list(self.edges.keys())

    def add_edge(self, vertex, weight=0):
        print(f'Adding vertex {vertex}')
        self.edges[vertex] = weight
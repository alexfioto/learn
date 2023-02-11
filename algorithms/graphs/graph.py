class Graph:
    def __init__(self, directed=False):
        self.directed = directed
        self.graph_dict = {}
    
    def add_vertex(self, vertex):
        print('Adding' + vertex.value)
        self.graph_dict[vertex.value] = vertex

    def add_edge(self, from_vertex, to_vertex, weight=0):
        print(f'Adding edge from {from_vertex} to {to_vertex}.')
        graph_dict[from_vertex.value].add_edge(to_vertex.value, weight)
        if not self.directed:
            graph_dict[to_vertex.value].add_edge(from_vertex.value, weight)

    def find_path(self, start_vertex, end_vertex):
        print(f'Searching from {start_vertex} to {end_vertex}')
        start = [start_vertex]
        seen = {}
        while start:
            current_vertex = start.pop(0)
            seen[current_vertex] = True
            if current_vertex == end_vertex:
                return True
            else:
                vertices_to_visit = set(self.graph_dict[current_vertex].edges.keys())
                start += [vertex for vertex in vertices_to_visit if vertex not in seen]
        return False




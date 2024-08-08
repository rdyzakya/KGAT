import networkx as nx
from collections import Counter
import json

year = 2024
path = f"./lmkbc/{year}/triples.json"

triples = json.load(open(path))

for i, el in enumerate(triples):
    el[1] = f"rel{el[1]}"
    triples[i] = tuple(el)

def create_graph_from_triples(triples):
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add edges to the graph
    for subject, relation, obj in triples:
        # Use (subject, object) pair as an edge with relation as an attribute
        G.add_edge(subject, obj, relation=relation)
    
    return G

def calculate_graph_statistics(G):
    # Calculate number of nodes
    num_nodes = G.number_of_nodes()
    
    # Calculate number of edges
    num_edges = G.number_of_edges()
    
    # Calculate degree distribution
    degree_sequence = [d for n, d in G.degree()]
    degree_count = Counter(degree_sequence)
    degrees, counts = zip(*degree_count.items())
    
    # Calculate average degree
    avg_degree = sum(degree_sequence) / num_nodes if num_nodes > 0 else 0
    
    # Calculate number of connected components (for undirected version)
    undirected_G = G.to_undirected()
    num_connected_components = nx.number_connected_components(undirected_G)
    
    # Calculate number of self-loops
    num_self_loops = nx.number_of_selfloops(G)
    
    # Calculate the density of the graph
    density = nx.density(G)
    
    # Calculate in-degree and out-degree distributions
    in_degree_sequence = [d for n, d in G.in_degree()]
    out_degree_sequence = [d for n, d in G.out_degree()]
    in_degree_count = Counter(in_degree_sequence)
    out_degree_count = Counter(out_degree_sequence)
    
    return {
        'Number of nodes': num_nodes,
        'Number of edges': num_edges,
        'Degree distribution': degree_count,
        'Average degree': avg_degree,
        'Number of connected components': num_connected_components,
        'Number of self-loops': num_self_loops,
        'Graph density': density,
        'In-degree distribution': in_degree_count,
        'Out-degree distribution': out_degree_count
    }

def main(triples):
    # Create the graph
    G = create_graph_from_triples(triples)
    
    # Calculate statistics
    stats = calculate_graph_statistics(G)

    with open(f"./lmkbc/{year}/stats.json", 'w') as fp:
        json.dump(stats, fp)
    
    # Print statistics
    for stat, value in stats.items():
        print(f"{stat}: {value}")

if __name__ == "__main__":
    main(triples)
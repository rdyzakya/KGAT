import os, json
import networkx as nx

def load_json(path):
    with open(path, 'r') as fp:
        data = json.load(fp)
    return data

def summarize_statistic(graph_data):
    G = nx.DiGraph()

    for s, o, r in graph_data["coo"]:
        G.add_edge(s, o, relation=r)
    
    stats = {
        "N Triples" : len(graph_data["coo"]),
        "N Nodes" : graph_data["num_entities"],
        "N Relation" : graph_data["num_relations"],
        "Avg node degree" : len(graph_data["coo"])/ graph_data["num_entities"],
        "Density" : nx.density(G),
        "Avg degree centrality": sum(nx.degree_centrality(G).values())/graph_data["num_entities"],
        "Avg betweenness centrality": sum(nx.betweenness_centrality(G).values())/graph_data["num_entities"],
        "Avg closeness centrality": sum(nx.closeness_centrality(G).values())/graph_data["num_entities"],
    }

    return stats

print("Processing statistics...")
listdir = os.listdir()
statistics = dict()
for path in listdir:
    if not os.path.isdir(path):
        continue
    statistics[path] = dict()
    proc_path = os.path.join(path, "proc")
    for fname in os.listdir(proc_path):
        graph_data = load_json(
            os.path.join(proc_path, fname)
        )
        statistics[path][fname.split('.')[0]] = summarize_statistic(graph_data)

with open("statistics.json", 'w') as fp:
    json.dump(statistics, fp)
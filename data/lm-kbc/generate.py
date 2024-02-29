import os
import subprocess
import shutil
import networkx as nx
import json

def run_command(command):
    try:
        result = subprocess.run(command, check=True, capture_output=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
    else:
        # Get the command output
        output = result.stdout.decode("utf-8")
        print(output)

def change_permissions(directory, permissions):
    """
    Attempts to change permissions of all files within a directory.

    Args:
        directory (str): Path to the directory.
        permissions (int): Octal representation of the desired permissions (e.g., 0o755).

    Raises:
        OSError: If an error occurs while changing permissions.
    """

    for root, dirs, files in os.walk(directory):
        for file in files:
            try:
                os.chmod(os.path.join(root, file), permissions)
            except OSError as e:
                print(f"Error changing permissions for {file}: {e}")

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

url_lmkbc2022 = "https://github.com/lm-kbc/dataset2022"
url_lmkbc2023 = "https://github.com/lm-kbc/dataset2023"
url_mars = "https://github.com/zjunlp/MKG_Analogy"

# lm-kbc 2022
print("Downloading and preprocessing lm-kbc2022...")
run_command(["git", "clone", url_lmkbc2022])
try:
    shutil.copytree("./dataset2022/data", "./lm-kbc2022/raw")
except:
    pass
change_permissions("./dataset2022/.git", 0o777)
shutil.rmtree("./dataset2022")
run_command(["cd", "lm-kbc2022", "&&", "python", "convert.py"])

# lm-kbc 2023
print("Downloading and preprocessing lm-kbc2023...")
run_command(["git", "clone", url_lmkbc2023])
try:
    shutil.copytree("./dataset2023/data", "./lm-kbc2023/raw")
except:
    pass
change_permissions("./dataset2023/.git", 0o777)
shutil.rmtree("./dataset2023")
run_command(["cd", "lm-kbc2023", "&&", "python", "convert.py"])

# mars
run_command(["git", "clone", url_mars])
print("Downloading and preprocessing mars...")
try:
    shutil.copytree("./MKG_Analogy/MarT/dataset/MarKG", "./mars/raw")
except:
    pass
change_permissions("./MKG_Analogy/.git", 0o777)
shutil.rmtree("./MKG_Analogy")
run_command(["cd", "mars", "&&", "python", "convert.py"])

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
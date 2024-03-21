import os
import subprocess
import shutil
import json
import requests
import tarfile, gzip, zipfile

def run_command(command):
    subprocess.run(command, check=True, capture_output=True, shell=True)

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
            os.chmod(os.path.join(root, file), permissions)

def download_file(url, destination):
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(destination, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
    print(f"File downloaded successfully: {destination}")

def untar_tgz(tgz_file, destination):
    with tarfile.open(tgz_file, 'r:gz') as tar:
        tar.extractall(destination)
    print(f"Files extracted successfully to: {destination}")

def extract_gzip(gzip_file, output_file):
    with gzip.open(gzip_file, 'rb') as f_in:
        with open(output_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def extract_zip(zip_file, destination):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(destination)
    print(f"Files extracted successfully to: {destination}")

def load_json(path):
    with open(path, 'r') as fp:
        data = json.load(fp)
    return data

url_atomic = "https://storage.googleapis.com/ai2-mosaic/public/atomic/v1.0/atomic_data.tgz"
url_graphwriter = "https://github.com/rikdz/GraphWriter"
url_qagnn = "https://nlp.stanford.edu/projects/myasu/QAGNN/data_preprocessed_release.zip"
url_text2kg = "https://github.com/cenguix/Text2KGBench"

# atomic
print("Downloading and preprocessing atomic...")
download_file(url_atomic, url_atomic.split('/')[-1])

if not os.path.exists("./atomic/raw"):
    os.makedirs("./atomic/raw")

untar_tgz(url_atomic.split('/')[-1], "./atomic/raw")

os.remove(url_atomic.split('/')[-1])

run_command(["cd", "atomic", "&&", "python", "convert.py"])

# graphwriter
print("Downloading and preprocessing graphwriter...")
run_command(["git", "clone", url_graphwriter])
try:
    shutil.copytree("./GraphWriter/data", "./graph-writer/raw")
except:
    pass
change_permissions("./GraphWriter/.git", 0o777)
shutil.rmtree("./GraphWriter")
run_command(["cd", "graph-writer", "&&", "python", "convert.py"])

# qagnn
print("Downloading and preprocessing qagnn...")
download_file(url_qagnn, url_qagnn.split('/')[-1])

if not os.path.exists("./qagnn/raw"):
    os.makedirs("./qagnn/raw")

extract_zip("data_preprocessed_release.zip", "./qagnn/raw")

os.remove(url_qagnn.split('/')[-1])

run_command(["cd", "qagnn", "&&", "python", "convert.py"])

# text2kg
print("Downloading and preprocessing text2kg...")
run_command(["git", "clone", url_text2kg])
try:
    shutil.copytree("./Text2KGBench/data", "./Text2KGBench/raw")
except:
    pass
change_permissions("./Text2KGBench/.git", 0o777)
shutil.rmtree("./Text2KGBench")
run_command(["cd", "text2kg", "&&", "python", "convert.py"])

# webnlg
print("Downloading and preprocessing webnlg...")
run_command(["cd", "webnlg", "&&", "python", "convert.py"])
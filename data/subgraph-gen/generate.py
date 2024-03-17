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
            try:
                os.chmod(os.path.join(root, file), permissions)
            except OSError as e:
                print(f"Error changing permissions for {file}: {e}")

def download_file(url, destination):
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            with open(destination, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
        print(f"File downloaded successfully: {destination}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")

def untar_tgz(tgz_file, destination):
    try:
        with tarfile.open(tgz_file, 'r:gz') as tar:
            tar.extractall(destination)
        print(f"Files extracted successfully to: {destination}")
    except tarfile.TarError as e:
        print(f"Error extracting files: {e}")

def extract_gzip(gzip_file, output_file):
    with gzip.open(gzip_file, 'rb') as f_in:
        with open(output_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def extract_zip(zip_file, destination):
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(destination)
        print(f"Files extracted successfully to: {destination}")
    except Exception as e:
        print(f"Error extracting files: {e}")

def load_json(path):
    with open(path, 'r') as fp:
        data = json.load(fp)
    return data

url_atomic = "https://storage.googleapis.com/ai2-mosaic/public/atomic/v1.0/atomic_data.tgz"
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

# qagnn
print("Downloading and preprocessing qagnn...")
download_file(url_qagnn, url_qagnn.split('/')[-1])

if not os.path.exists("./qagnn/raw"):
    os.makedirs("./qagnn/raw")

extract_zip("data_preprocessed_release.zip", "./qagnn/raw")

os.remove(url_qagnn.split('/')[-1])

run_command(["cd", "qagnn", "&&", "python", "convert.py"])

# # lm-kbc 2023
# print("Downloading and preprocessing lm-kbc2023...")
# run_command(["git", "clone", url_lmkbc2023])
# try:
#     shutil.copytree("./dataset2023/data", "./lm-kbc2023/raw")
# except:
#     pass
# change_permissions("./dataset2023/.git", 0o777)
# shutil.rmtree("./dataset2023")
# run_command(["cd", "lm-kbc2023", "&&", "python", "convert.py"])

# # mars
# run_command(["git", "clone", url_mars])
# print("Downloading and preprocessing mars...")
# try:
#     shutil.copytree("./MKG_Analogy/MarT/dataset/MarKG", "./mars/raw")
# except:
#     pass
# change_permissions("./MKG_Analogy/.git", 0o777)
# shutil.rmtree("./MKG_Analogy")
# run_command(["cd", "mars", "&&", "python", "convert.py"])
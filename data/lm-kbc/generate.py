import os
import subprocess
import shutil
import json
import requests
import gzip, zipfile

def run_command(command):
    result = subprocess.run(command, check=True, capture_output=False, shell=True)

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

url_lmkbc2022 = "https://github.com/lm-kbc/dataset2022"
url_lmkbc2023 = "https://github.com/lm-kbc/dataset2023"
url_mars = "https://github.com/zjunlp/MKG_Analogy"

url_freebase = "https://download.microsoft.com/download/8/7/0/8700516A-AB3D-4850-B4BB-805C515AECE1/FB15K-237.2.zip"
url_freebase_mid2name = "https://drive.usercontent.google.com/download?id=0B52yRXcdpG6MaHA5ZW9CZ21MbVk&export=download&authuser=0&confirm=t&uuid=9f09fae2-0885-4aff-bf65-e55c9ebeac19&at=APZUnTW9rF2jBHGa2Iqq5_zrX7kv%3A1709103664110"

url_conceptnet = "https://nlp.stanford.edu/projects/myasu/QAGNN/data_preprocessed_release.zip"
# # lm-kbc 2022
# print("Downloading and preprocessing lm-kbc2022...")
# run_command(["git", "clone", url_lmkbc2022])
# try:
#     shutil.copytree("./dataset2022/data", "./lm-kbc2022/raw")
# except:
#     pass
# change_permissions("./dataset2022/.git", 0o777)
# shutil.rmtree("./dataset2022")
# run_command(["cd", "lm-kbc2022", "&&", "python", "convert.py"])

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

# # freebase
# print("Downloading and preprocessing freebase...")
# download_file(url_freebase, url_freebase.split('/')[-1])
# download_file(url_freebase_mid2name, "mid2name.gz")

# if not os.path.exists("./freebase/raw"):
#     os.makedirs("./freebase/raw")

# extract_zip(url_freebase.split('/')[-1], "./freebase/raw")
# extract_gzip("mid2name.gz", "./freebase/raw/mid2name.tsv")

# os.remove(url_freebase.split('/')[-1])
# os.remove("mid2name.gz")

# run_command(["cd", "freebase", "&&", "python", "convert.py"])

# # conceptnet
# print("Downloading and preprocessing conceptnet...")
# download_file(url_conceptnet, url_conceptnet.split('/')[-1])

# if not os.path.exists("./conceptnet/raw"):
#     os.makedirs("./conceptnet/raw")

# extract_zip("data_preprocessed_release.zip", "./conceptnet/raw")

# os.remove(url_conceptnet.split('/')[-1])

run_command(["cd", "conceptnet", "&&", "python", "convert.py"])
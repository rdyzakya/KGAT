def read_txt(path):
    with open(path, 'r', encoding="utf-8") as fp:
        data = fp.read().strip().splitlines()
    return data
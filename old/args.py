from argparse import ArgumentParser

def train_sg_args():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", type=str, help="model config json path", 
                        default="./config/model/default.json")
    parser.add_argument("-t", "--train", type=str, help="train config json path",
                        default="./config/train/sg-default.json")
    parser.add_argument("--data", type=str, help="Data directory",
                        default="./data/subgraph-gen/atomic/proc")
    parser.add_argument("--gpu", type=str, help="GPU index",
                        default="0")
    args = parser.parse_args()
    return args
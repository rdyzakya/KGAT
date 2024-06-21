from argparse import ArgumentParser

def sg_train():
    parser = ArgumentParser()
    
    parser.add_argument("--gpu", type=str, help="GPU device id", default="0")
    parser.add_argument("--data", type=str, help="Data directory", default="./data/subgraph-gen/atomic/proc")
    parser.add_argument("--model", type=str, help="Model config path", default="./config/model/default.json")
    parser.add_argument("--epoch", type=int, help="Number of epoch", default=10)
    parser.add_argument("--lr", type=float, help="Learning rate", default=1e-3)
    parser.add_argument("--bsize", type=int, help="Batch size", default=8)
    parser.add_argument("--hsbsize", type=int, help="Hidden state batch size", default=32)
    parser.add_argument("--out", type=str, help="Output directory", default="./out")
    parser.add_argument("--mcp", type=int, help="Max number of checkpoint", default=3)
    parser.add_argument("--best_metrics", type=str, help="Best metrics option", default="sg_loss")
    parser.add_argument("--load_best", action="store_true", help="Load best model at the end")
    parser.add_argument("--optim", type=str, help="Optimizer type", default="adam")
    parser.add_argument("--n_data_train", type=int, help="First n data in training data (for debug)")
    parser.add_argument("--n_data_val", type=int, help="First n data in validation data (for debug)")
    parser.add_argument("--seed", type=int, help="Random seed initialization", default=42)
    parser.add_argument("--nlw", type=str, help="Negative loss weight (float or 'auto')", default="1.0")
    parser.add_argument("--alpha", type=float, help="Alpha term", default=1.0)

    args = parser.parse_args()
    return args
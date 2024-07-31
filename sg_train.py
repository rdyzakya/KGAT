from argparse import ArgumentParser
import os
def init_args():
    parser = ArgumentParser()
    # DATA RELATED
    parser.add_argument("--data-dir", type=str, help="Data directory", default="./data/subgraph-gen/webnlg")
    parser.add_argument("--super-set", action="store_true", help="Train super set")
    parser.add_argument("--n-ref-min", type=int, help="N reference min", default=10)
    parser.add_argument("--n-ref-max", type=int, help="N reference max", default=50)
    parser.add_argument("--stay-ratio-min", type=float, help="Stay ratio min", default=1.0)
    parser.add_argument("--stay-ratio-max", type=float, help="Stay ratio max", default=1.0)
    parser.add_argument("--save-items", action="store_true", help="Save items (jsonl)")
    parser.add_argument("--load_items", action="store_true", help="Load items")
    parser.add_argument("--sentence-emb-mode", type=str, help="Sentence embedding mode", default="baseline")
    parser.add_argument("--lm", type=str, help="HF lm model name or path", default="openai-community/gpt2")
    parser.add_argument("--sentence-emb-idx", type=int, help="Sentence embedding index (layer index)")
    parser.add_argument("--alias-idx", type=int, help="Alias index (some entity have several aliases, affect the entity node attribute)")

    # MODEL RELATED
    parser.add_argument("--hidden", type=int, help="Hidden channel dimension") # based on gatv2 paper, for text embedding features, using 2 * input dim, but remember there is head
    parser.add_argument("--layer", type=int, help="Number of layers (1..)", default=2) # based on VGAE paper and training details in GATv2
    parser.add_argument("--head", type=int, help="Number of attention heads", default=1) # based on gatv2 paper

    # TRAINING RELATED
    parser.add_argument("--epoch", type=int, help="Epoch", default=10)
    parser.add_argument("--bsize", type=int, help="Batch size", default=8)
    parser.add_argument("--lr", type=float, help="Learning rate", default=1e-3) # based on default adam, also mentioned in gatv2 paper
    parser.add_argument("--decay", type=float, help="Weight decay", default=0.5) # based on gatv2 paper
    parser.add_argument("--all", action="store_true", help="Return all score in the adjacency matrix if set, else only the links from edge_index")
    # parser.add_argument("--estop", action="store_true", help="Early stopping")
    parser.add_argument("--out", type=str, help="Out dir", default="./out")

    parser.add_argument("--seed", type=int, help="Random seed", default=42)
    parser.add_argument("--gpu", type=str, help="Gpu ids")

    args = parser.parse_args()

    return args

args = init_args()
if args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from torch_geometric import seed_everything
from data import DSBuilder, SubgraphGenDataset, SubgraphGenCollator
from torch.utils.data import DataLoader
from model import MultiheadGAE
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report
import time

def create_adj_label(n_node, n_relation, edge_index, link_label):
    adj = torch.zeros(n_relation, n_node, n_node).float()
    true_edge_index = edge_index[:,link_label.bool()]
    adj[true_edge_index[1], true_edge_index[0], true_edge_index[2]] = 1.0
    return adj

if __name__ == "__main__":
    seed_everything(args.seed)
    ## DATASET
    train_builder = DSBuilder(
        triples_path=os.path.join(args.data_dir, "triples.json"),
        data_path=os.path.join(args.data_dir, "all.jsonl" if args.super_set else "train.jsonl"),
        n_reference_min=args.n_ref_min,
        n_reference_max=args.n_ref_max,
        stay_ratio_min=args.stay_ratio_min,
        stay_ratio_max=args.stay_ratio_max,
        random_state=args.seed,
        n_pick=1,
        items_path="./train-items.jsonl",
        save_items=bool(args.save_items),
        load=bool(args.load_items)
    )

    val_builder = DSBuilder(
        triples_path=os.path.join(args.data_dir, "triples.json"),
        data_path=os.path.join("dev.jsonl"),
        n_reference_min=args.n_ref_min,
        n_reference_max=args.n_ref_max,
        stay_ratio_min=args.stay_ratio_min,
        stay_ratio_max=args.stay_ratio_max,
        random_state=args.seed,
        n_pick=1,
        items_path="./dev-items.jsonl",
        save_items=bool(args.save_items),
        load=bool(args.load_items)
    )

    texts_tensor_path = os.path.join(args.data_dir, f"texts.{args.lm.replace('/', '_')}.{args.sentence_emb_idx}.tensor")
    texts_tensor_path = texts_tensor_path if os.path.exists(texts_tensor_path) else os.path.join(args.data_dir, f"texts.{args.lm.replace('/', '_')}.tensor")

    entities_tensor_path = os.path.join(args.data_dir, f"entities.{args.lm.replace('/', '_')}.{args.sentence_emb_idx}.tensor")
    entities_tensor_path = entities_tensor_path if os.path.exists(entities_tensor_path) else os.path.join(args.data_dir, f"entities.{args.lm.replace('/', '_')}.tensor")

    relations_tensor_path = os.path.join(args.data_dir, f"relations.{args.lm.replace('/', '_')}.{args.sentence_emb_idx}.tensor")
    relations_tensor_path = relations_tensor_path if os.path.exists(relations_tensor_path) else os.path.join(args.data_dir, f"relations.{args.lm.replace('/', '_')}.tensor")

    train_ds = SubgraphGenDataset(
        train_builder,
        os.path.join(args.data_dir, "texts.txt"),
        os.path.join(args.data_dir, "entities.txt"),
        os.path.join(args.data_dir, "relations.txt"),
        os.path.join(args.data_dir, "entitias_alias.jsonl"),
        texts_tensor_path=texts_tensor_path,
        entities_tensor_path=entities_tensor_path,
        relations_tensor_path=relations_tensor_path,
        sentence_emb_mode=args.sentence_emb_mode,
        sentence_emb_index=args.sentence_emb_idx
    )

    val_ds = SubgraphGenDataset(
        val_builder,
        os.path.join(args.data_dir, "texts.txt"),
        os.path.join(args.data_dir, "entities.txt"),
        os.path.join(args.data_dir, "relations.txt"),
        os.path.join(args.data_dir, "entitias_alias.jsonl"),
        texts_tensor_path=texts_tensor_path,
        entities_tensor_path=entities_tensor_path,
        relations_tensor_path=relations_tensor_path,
        sentence_emb_mode=args.sentence_emb_mode,
        sentence_emb_index=args.sentence_emb_idx
    )

    train_ds.prepare_train()
    val_ds.prepare_eval()

    train_collator = SubgraphGenCollator(train_ds, alias_idx=args.alias_idx)
    val_collator = SubgraphGenCollator(val_ds, alias_idx=args.alias_idx)

    train_dataloader = DataLoader(train_ds, batch_size=args.bsize, shuffle=True, collate_fn=train_collator)
    val_dataloader = DataLoader(val_ds, batch_size=args.bsize, shuffle=False, collate_fn=val_collator)
    
    ## MODEL
    n_features = train_ds.entities_attr.shape[1]
    hidden_channels = args.hidden or 2 * n_features
    model = MultiheadGAE(
        in_channels=train_ds.entities_attr.shape[1], 
        hidden_channels=hidden_channels, 
        num_layers=args.layer, 
        heads=args.head, 
        out_channels=None, 
        negative_slope=0.2, # based on the gat v1 paper
        dropout=0.0, 
        add_self_loops=True, 
        bias=True, 
        share_weights=False,
        subgraph=True,
    )
    
    ## TRAIN LOOP
    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_bar = tqdm(total=args.epoch*len(train_dataloader), desc="Training")

    history = []
    for e in range(args.epoch):
        entry = {
            "epoch" : e + 1
        }
        train_start_time = time.time()
        model.train()
        
        out_node = []
        out_link = []

        label_node = []
        label_link = []
        for batch in train_dataloader:
            for k, v in batch.items():
                batch[k] = v.to(device)
            link_cls_label = batch.pop("link_cls_label")
            node_cls_label = batch.pop("node_cls_label")

            z, all_adj, all_alpha, out_link, out_node = model.forward(all=args.all, 
                                                                      sigmoid=False, 
                                                                      return_attention_weights=False, 
                                                                      **batch)
            
            out_node = out_node.view(-1)
            out_link = out_link.view(-1)

            node_cls_label = node_cls_label.float().view(-1)
            link_cls_label = create_adj_label(batch["x"].shape[0], batch["relations"].shape[0], batch["edge_index"], link_label=link_cls_label).view(-1) if args.all else link_cls_label.float().view(-1)

            loss_node = criterion(out_node, 
                                  node_cls_label)
            loss_link = criterion(
                out_link,
                link_cls_label
            )

            loss = loss_node + loss_link

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_bar.update()

            out_node.append(out_node.detach().cpu())
            out_link.append(out_link.detach().cpu())

            label_node.append(node_cls_label.cpu())
            label_link.append(link_cls_label.cpu())
        
        train_end_time = time.time()

        entry["train_time"] = train_end_time - train_start_time

        out_node = torch.cat(out_node)
        out_link = torch.cat(out_link)

        label_node = torch.cat(label_node)
        label_link = torch.cat(label_link)

        train_node_loss = criterion(out_node, label_node).item()
        train_link_loss = criterion(out_link, label_link).item()

        entry["train_node_loss"] = train_node_loss
        entry["train_link_loss"] = train_link_loss

        # compute metrics
        report_train_node = classification_report(
            y_pred=out_node.round(),
            y_true=label_node.int(),
            output_dict=True
        )
        report_train_link = classification_report(
            y_pred=out_link.round(),
            y_true=label_link.int(),
            output_dict=True
        )

        entry["train_node_accuracy"] = report_train_node["accuracy"]
        entry["train_node_precision"] = report_train_node["macro avg"]["precision"]
        entry["train_node_recall"] = report_train_node["macro avg"]["recall"]
        entry["train_node_f1"] = report_train_node["macro"]["f1-score"]

        entry["train_link_accuracy"] = report_train_link["accuracy"]
        entry["train_link_precision"] = report_train_link["macro avg"]["precision"]
        entry["train_link_recall"] = report_train_link["macro avg"]["recall"]
        entry["train_link_f1"] = report_train_link["macro"]["f1-score"]

        val_start_time = time.time()
        model.eval()
        val_bar = tqdm(total=len(val_dataloader), desc="Evaluation")
        with torch.no_grad():
            out_node = []
            out_link = []

            label_node = []
            label_link = []
            for batch in val_dataloader:
                for k, v in batch.items():
                    batch[k] = v.to(device)
                link_cls_label = batch.pop("link_cls_label")
                node_cls_label = batch.pop("node_cls_label")

                z, all_adj, all_alpha, out_link, out_node = model.forward(all=args.all, 
                                                                      sigmoid=False, 
                                                                      return_attention_weights=False, 
                                                                      **batch)
            
                out_node = out_node.view(-1)
                out_link = out_link.view(-1)

                node_cls_label = node_cls_label.float().view(-1)
                link_cls_label = create_adj_label(batch["x"].shape[0], batch["relations"].shape[0], batch["edge_index"], link_label=link_cls_label).view(-1) if args.all else link_cls_label.float().view(-1)

                out_node.append(out_node.detach().cpu())
                out_link.append(out_link.detach().cpu())

                label_node.append(node_cls_label.cpu())
                label_link.append(link_cls_label.cpu())

                val_bar.update()
        val_end_time = time.time()

        out_node = torch.cat(out_node)
        out_link = torch.cat(out_link)

        label_node = torch.cat(label_node)
        label_link = torch.cat(label_link)

        val_node_loss = criterion(out_node, label_node).item()
        val_link_loss = criterion(out_link, label_link).item()

        entry["val_node_loss"] = val_node_loss
        entry["val_link_loss"] = val_link_loss

        # compute metrics
        report_val_node = classification_report(
            y_pred=out_node.round(),
            y_true=label_node.int(),
            output_dict=True
        )
        report_val_link = classification_report(
            y_pred=out_link.round(),
            y_true=label_link.int(),
            output_dict=True
        )

        entry["val_node_accuracy"] = report_val_node["accuracy"]
        entry["val_node_precision"] = report_val_node["macro avg"]["precision"]
        entry["val_node_recall"] = report_val_node["macro avg"]["recall"]
        entry["val_node_f1"] = report_val_node["macro"]["f1-score"]

        entry["val_link_accuracy"] = report_val_link["accuracy"]
        entry["val_link_precision"] = report_val_link["macro avg"]["precision"]
        entry["val_link_recall"] = report_val_link["macro avg"]["recall"]
        entry["val_link_f1"] = report_val_link["macro"]["f1-score"]
        history.append(entry)
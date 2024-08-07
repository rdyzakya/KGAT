from argparse import ArgumentParser
import os
import sys
sys.path.append("./src")

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
    parser.add_argument("--load-items", action="store_true", help="Load items")
    parser.add_argument("--sentence-emb-mode", type=str, help="Sentence embedding mode", default="baseline")
    parser.add_argument("--lm", type=str, help="HF lm model name or path", default="openai-community/gpt2")
    parser.add_argument("--sentence-emb-idx", type=int, help="Sentence embedding index (layer index)")
    parser.add_argument("--alias-idx", type=int, help="Alias index (some entity have several aliases, affect the entity node attribute)")
    parser.add_argument("--n-token", type=int, default=1)

    # TRAINING RELATED
    parser.add_argument("--epoch", type=int, help="Epoch", default=10)
    parser.add_argument("--bsize", type=int, help="Batch size", default=8)
    parser.add_argument("--lr", type=float, help="Learning rate", default=1e-3) # based on default adam, also mentioned in unimp paper
    parser.add_argument("--decay", type=float, help="Weight decay", default=0.0005) # based on unimp paper
    
    parser.add_argument("--estop", action="store_true", help="Perform early stopping")
    parser.add_argument("--estop-patience", type=int, help="Early stopping patience", default=3)
    parser.add_argument("--estop-delta", type=float, help="Early stopping delta", default=0.05)
    parser.add_argument("--best-metrics", type=str, help="Early stopping metrics", default="f1")

    parser.add_argument("--load-best", action="store_true", help="Load best at end")
    parser.add_argument("--max-ckpt", type=int, help="Max checkpoint", default=5)

    parser.add_argument("--test", action="store_true", help="Perform final evaluation (test)")

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
from model import KGATModel
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report
import time
import utils
import json

def create_adj_label(n_node, n_relation, edge_index, link_label):
    adj = torch.zeros(n_relation, n_node, n_node).float()
    true_edge_index = edge_index[:,link_label.bool()]
    adj[true_edge_index[1], true_edge_index[0], true_edge_index[2]] = 1.0
    return adj

def loop(model, dataloader, device, args, optimizer, criterion, pbar, val=False): # train/val loop
    entry = {}
    start_time = time.time()
    if val:
        model.eval()
    else:
        model.train()

    qr_out = []
    qr_labels = []
    rv_out = []
    qv_out = []
    for batch in dataloader:
        for k, v in batch.items():
            batch[k] = v.to(device)
        link_cls_label = batch.pop("link_cls_label")
        node_cls_label = batch.pop("node_cls_label")

        link_cls_label = link_cls_label.float()

        if val:
            with torch.no_grad():
                out = model.forward(sigmoid=False, 
                                    allow_intersection=False, 
                                    **batch)
        else:
            out = model.forward(sigmoid=False, 
                                allow_intersection=False, 
                                **batch)
        
        if len(out) > 1:
            query_reference_out, reference_values_out, query_values_out = out
            reference_values_out = reference_values_out.view(-1)
            query_values_out= query_values_out.view(-1)
        else:
            query_reference_out = out
        
        query_reference_out = query_reference_out.view(-1)

        if not val:
            qr_loss = criterion(
                query_reference_out,
                link_cls_label
            )

            # kl_loss = model.vgae.kl_loss()

            loss = qr_loss #+ kl_loss

            if len(out) > 1:
                rv_loss = criterion(
                    reference_values_out,
                    link_cls_label
                )

                qv_loss = criterion(
                    query_values_out,
                    torch.ones_like(query_values_out)
                )

                loss = loss + rv_loss + qv_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        pbar.update()

        qr_out.append(query_reference_out.detach().cpu())
        qr_labels.append(link_cls_label.detach().cpu())

        if len(out) > 1:
            rv_out.append(reference_values_out.detach().cpu())
            qv_out.append(query_values_out.detach().cpu())
    
    end_time = time.time()

    entry["time"] = end_time - start_time

    qr_out = torch.cat(qr_out)
    qr_labels = torch.cat(qr_labels)

    qr_report = classification_report(
        y_pred=qr_out.sigmoid().round(),
        y_true=qr_labels.int(),
        output_dict=True
    )

    entry["accuracy"] = qr_report["accuracy"]
    entry["precision"] = qr_report["macro avg"]["precision"]
    entry["recall"] = qr_report["macro avg"]["recall"]
    entry["f1"] = qr_report["macro avg"]["f1-score"]
    entry["qr_loss"] = criterion(qr_out, qr_labels).item()

    if len(rv_out) > 0 and len(qv_out) > 0:
        rv_out = torch.cat(rv_out)
        qv_out = torch.cat(qv_out)
        entry["rv_loss"] = criterion(rv_out, qr_labels).item()
        entry["qv_loss"] = criterion(qv_out, torch.ones_like(qv_out)).item()

    return entry

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
        data_path=os.path.join(args.data_dir, "dev.jsonl"),
        n_reference_min=args.n_ref_min,
        n_reference_max=args.n_ref_max,
        stay_ratio_min=1.0,
        stay_ratio_max=1.0,
        random_state=args.seed,
        n_pick=1,
        items_path="./dev-items.jsonl",
        save_items=bool(args.save_items),
        load=bool(args.load_items)
    )

    texts_tensor_path = os.path.join(args.data_dir, f"texts.{args.lm.replace('/', '_')}.n_token={args.n_token}.index={args.sentence_emb_idx}.tensor")
    texts_tensor_path = texts_tensor_path if os.path.exists(texts_tensor_path) else os.path.join(args.data_dir, f"texts.{args.lm.replace('/', '_')}.n_token={args.n_token}.tensor")

    entities_tensor_path = os.path.join(args.data_dir, f"entities.{args.lm.replace('/', '_')}.n_token={args.n_token}.index={args.sentence_emb_idx}.tensor")
    entities_tensor_path = entities_tensor_path if os.path.exists(entities_tensor_path) else os.path.join(args.data_dir, f"entities.n_token={args.n_token}.tensor")

    relations_tensor_path = os.path.join(args.data_dir, f"relations.{args.lm.replace('/', '_')}.n_token={args.n_token}.index={args.sentence_emb_idx}.tensor")
    relations_tensor_path = relations_tensor_path if os.path.exists(relations_tensor_path) else os.path.join(args.data_dir, f"relations.n_token={args.n_token}.tensor")

    train_ds = SubgraphGenDataset(
        train_builder,
        os.path.join(args.data_dir, "texts.txt"),
        os.path.join(args.data_dir, "entities.txt"),
        os.path.join(args.data_dir, "relations.txt"),
        os.path.join(args.data_dir, "entities_alias.jsonl"),
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
        os.path.join(args.data_dir, "entities_alias.jsonl"),
        texts_tensor_path=None,
        entities_tensor_path=None,
        relations_tensor_path=None,
        sentence_emb_mode=args.sentence_emb_mode,
        sentence_emb_index=args.sentence_emb_idx
    )

    val_ds.texts_attr = train_ds.texts_attr
    val_ds.entities_attr = train_ds.entities_attr
    val_ds.relations_attr = train_ds.relations_attr

    train_ds.prepare_train()
    val_ds.prepare_eval()

    train_collator = SubgraphGenCollator(train_ds, alias_idx=args.alias_idx)
    val_collator = SubgraphGenCollator(val_ds, alias_idx=args.alias_idx)

    train_dataloader = DataLoader(train_ds, batch_size=args.bsize, shuffle=False, collate_fn=train_collator)
    val_dataloader = DataLoader(val_ds, batch_size=args.bsize, shuffle=False, collate_fn=val_collator)
    
    ## MODEL
    model = KGATModel(d_model=train_ds.entities_attr.shape[1],
                    d_ff=args.d_ff,
                    heads=args.head,
                    beta=args.beta,
                    dropout=args.dropout,
                    bias=args.bias,
                    transform_relation=args.relation,
                    num_block=args.n_block,)
    
    ## TRAIN LOOP
    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    metrics_name = "val_" + args.best_metrics if not args.best_metrics.startswith("val_") else args.best_metrics
    greater_is_better = False if "loss" in metrics_name else True
    
    early_stopper = utils.EarlyStopper(args.estop_patience, args.estop_delta, greater_is_better)
    saveload = utils.SaveAndLoad(model, args.out, metrics_name, args.max_ckpt, greater_is_better)

    train_bar = tqdm(total=args.epoch*len(train_dataloader), desc="Training")

    history = []
    for e in range(args.epoch):
        entry = {"epoch" : e+1}
        
        train_entry = loop(model, train_dataloader, device, args, optimizer, criterion, train_bar, val=False)
        for k, v in train_entry.items():
            entry[f"train_{k}"] = v
        
        val_bar = tqdm(total=len(val_dataloader), desc="Val")
        val_entry = loop(model, val_dataloader, device, args, None, criterion, val_bar, val=True)
        for k, v in val_entry.items():
            entry[f"val_{k}"] = v
        
        print(entry)

        history.append(entry)
        saveload.save(history, is_ckpt=True)

    saveload.load_best(history)
    saveload.save(history, is_ckpt=False)

    ## EVALUATION
    if args.test:
        test_builder = DSBuilder(
            triples_path=os.path.join(args.data_dir, "triples.json"),
            data_path=os.path.join(args.data_dir, "test.jsonl"),
            n_reference_min=args.n_ref_min,
            n_reference_max=args.n_ref_max,
            stay_ratio_min=1.0,
            stay_ratio_max=1.0,
            random_state=args.seed,
            n_pick=1,
            items_path="./test-items.jsonl",
            save_items=bool(args.save_items),
            load=bool(args.load_items)
        )

        test_ds = SubgraphGenDataset(
            test_builder,
            os.path.join(args.data_dir, "texts.txt"),
            os.path.join(args.data_dir, "entities.txt"),
            os.path.join(args.data_dir, "relations.txt"),
            os.path.join(args.data_dir, "entities_alias.jsonl"),
            texts_tensor_path=None,
            entities_tensor_path=None,
            relations_tensor_path=None,
            sentence_emb_mode=args.sentence_emb_mode,
            sentence_emb_index=args.sentence_emb_idx
        )

        test_ds.texts_attr = train_ds.texts_attr
        test_ds.entities_attr = train_ds.entities_attr
        test_ds.relations_attr = train_ds.relations_attr

        test_ds.prepare_eval()

        test_collator = SubgraphGenCollator(test_ds, alias_idx=args.alias_idx)
        test_dataloader = DataLoader(test_ds, batch_size=args.bsize, shuffle=False, collate_fn=test_collator)

        test_bar = tqdm(total=len(test_dataloader), desc="Test")
        test_result = loop(model, test_dataloader, device, args, None, criterion, test_bar, val=True)

        with open(os.path.join(args.out, "test_metrics.json"), 'w') as fp:
            json.dump(test_result, fp)
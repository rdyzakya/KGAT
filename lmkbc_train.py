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
    parser.add_argument("--lm", type=str, help="HF lm model name or path", default="openai-community/gpt2") # and model
    parser.add_argument("--sentence-emb-idx", type=int, help="Sentence embedding index (layer index)")
    parser.add_argument("--alias-idx", type=int, help="Alias index (some entity have several aliases, affect the entity node attribute)")
    parser.add_argument("--prompt-idx", type=int, help="Prompt index")
    parser.add_argument("--n-token", type=int, default=1)

    # MODEL
    parser.add_argument("--kgat", type=str, help="Model path", required=True)

    # TRAINING RELATED
    parser.add_argument("--freeze-kgat", action="store_true")
    parser.add_argument("--first-epoch", type=int, help="First epoch", default=5)
    parser.add_argument("--second-epoch", type=int, help="Second epoch", default=10)
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
from data import DSBuilder, LMKBCDataset, LMKBCCollator
from torch.utils.data import DataLoader
from model import KGATModel, AutoModelForLMKBC, GraphPrefix, Pipeline
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report
import time
import utils
import json
from disambiguation import my_disambiguation

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
    ## TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(args.lm)
    tokenizer = utils.prepare_tokenizer(tokenizer)
    
    ## DATASET
    train_builder = DSBuilder(
        triples_path=os.path.join(args.data_dir, "triples.json"),
        data_path=os.path.join(args.data_dir, "all.jsonl" if args.super_set else "train.jsonl"),
        n_reference_min=args.n_ref_min,
        n_reference_max=args.n_ref_max,
        stay_ratio_min=0.0,
        stay_ratio_max=0.0,
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
        stay_ratio_min=0.0,
        stay_ratio_max=0.0,
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

    train_ds = LMKBCDataset(
        train_builder,
        os.path.join(args.data_dir, "texts.txt"),
        os.path.join(args.data_dir, "entities.txt"),
        os.path.join(args.data_dir, "relations.txt"),
        os.path.join(args.data_dir, "entities_alias.jsonl"),
        n_tokens=args.n_token,
        tokenizer=tokenizer,
        texts_tensor_path=texts_tensor_path,
        entities_tensor_path=entities_tensor_path,
        relations_tensor_path=relations_tensor_path,
        sentence_emb_mode=args.sentence_emb_mode,
        sentence_emb_index=args.sentence_emb_idx
    )

    val_ds = LMKBCDataset(
        val_builder,
        os.path.join(args.data_dir, "texts.txt"),
        os.path.join(args.data_dir, "entities.txt"),
        os.path.join(args.data_dir, "relations.txt"),
        os.path.join(args.data_dir, "entities_alias.jsonl"),
        n_tokens=args.n_token,
        tokenizer=tokenizer,
        texts_tensor_path=None,
        entities_tensor_path=None,
        relations_tensor_path=None,
        sentence_emb_mode=args.sentence_emb_mode,
        sentence_emb_index=args.sentence_emb_idx
    )

    val_ds.texts_attr = train_ds.texts_attr
    val_ds.entities_attr = train_ds.entities_attr
    val_ds.relations_attr = train_ds.relations_attr

    train_ds.prepare_train(prompt_idx=args.prompt_idx)
    val_ds.prepare_eval(prompt_idx=0)

    train_collator = LMKBCCollator(train_ds, tokenizer, alias_idx=args.alias_idx)
    val_collator = LMKBCCollator(val_ds, tokenizer, alias_idx=args.alias_idx)

    train_dataloader = DataLoader(train_ds, batch_size=args.bsize, shuffle=True, collate_fn=train_collator)
    val_dataloader = DataLoader(val_ds, batch_size=args.bsize, shuffle=False, collate_fn=val_collator)
    
    ## MODEL
    kgat_model = KGATModel.load(args.kgat)
    if args.freeze_kgat:
        kgat_model.freeze()

    language_model = AutoModelForLMKBC.from_pretrained(args.lm, device_map="auto")
    language_model = utils.prepare_model(language_model, tokenizer)
    language_model.freeze()
    
    graph_prefix = GraphPrefix(in_channels=train_ds.texts_attr.shape[1], d_model=language_model.embed_dim, n_token=args.n_token, bias=args.bias)

    pipe = Pipeline(kgat_model=kgat_model, graph_prefix=graph_prefix, language_model=language_model)
    
    ## TRAIN LOOP
    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pipe.kgat_model.to(device)
    pipe.graph_prefix.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(pipe.parameters(), lr=args.lr)

    metrics_name = "val_" + args.best_metrics if not args.best_metrics.startswith("val_") else args.best_metrics
    greater_is_better = False if "loss" in metrics_name else True
    
    early_stopper = utils.EarlyStopper(args.estop_patience, args.estop_delta, greater_is_better)
    saveload1 = utils.SaveAndLoad(pipe.kgat_model, os.path.join(args.out, "kgat", "first"), metrics_name, args.max_ckpt, greater_is_better)
    saveload2 = utils.SaveAndLoad(pipe.graph_prefix, os.path.join(args.out, "graph_prefix", "first"), metrics_name, args.max_ckpt, greater_is_better)

    train_bar = tqdm(total=args.first_epoch*len(train_dataloader), desc="Training")

    history = []
    for e in range(args.first_epoch):
        entry = {"epoch" : e+1}
        
        train_entry = loop(pipe, train_dataloader, device, args, optimizer, criterion, train_bar, val=False)
        for k, v in train_entry.items():
            entry[f"train_{k}"] = v
        
        val_bar = tqdm(total=len(val_dataloader), desc="Val")
        val_entry = loop(pipe, val_dataloader, device, args, None, criterion, val_bar, val=True)
        for k, v in val_entry.items():
            entry[f"val_{k}"] = v
        
        print(entry)

        history.append(entry)
        saveload1.save(history, is_ckpt=True)
        saveload2.save(history, is_ckpt=True)

    ## AUGMENT
    train_ds.prepare_augment(prompt_idx=0)

    augment_collator = LMKBCCollator(train_ds, tokenizer, alias_idx=args.alias_idx)
    augment_dataloader = DataLoader(train_ds, batch_size=args.bsize, shuffle=False, collate_fn=augment_collator)

    aug_bar = tqdm(total=len(augment_dataloader), desc="Augmentation")
    aug_entry = loop(pipe, augment_dataloader, device, args, None, criterion, aug_bar, val=True)

    predictions = aug_entry["predictions"]

    train_ds.augment(predictions)

    ## SECOND PHASE TRAIN
    train_ds.prepare_train(prompt_idx=args.prompt_idx)

    train_collator = LMKBCCollator(train_ds, tokenizer, alias_idx=args.alias_idx)
    val_collator = LMKBCCollator(val_ds, tokenizer, alias_idx=args.alias_idx)

    train_dataloader = DataLoader(train_ds, batch_size=args.bsize, shuffle=True, collate_fn=train_collator)
    val_dataloader = DataLoader(val_ds, batch_size=args.bsize, shuffle=False, collate_fn=val_collator)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(pipe.parameters(), lr=args.lr)

    metrics_name = "val_" + args.best_metrics if not args.best_metrics.startswith("val_") else args.best_metrics
    greater_is_better = False if "loss" in metrics_name else True
    
    early_stopper = utils.EarlyStopper(args.estop_patience, args.estop_delta, greater_is_better)
    saveload1 = utils.SaveAndLoad(pipe.kgat_model, os.path.join(args.out, "kgat", "second"), metrics_name, args.max_ckpt, greater_is_better)
    saveload2 = utils.SaveAndLoad(pipe.graph_prefix, os.path.join(args.out, "graph_prefix", "second"), metrics_name, args.max_ckpt, greater_is_better)

    train_bar = tqdm(total=args.second_epoch*len(train_dataloader), desc="Training")

    history = []
    for e in range(args.second_epoch):
        entry = {"epoch" : e+1}
        
        train_entry = loop(pipe, train_dataloader, device, args, optimizer, criterion, train_bar, val=False)
        for k, v in train_entry.items():
            entry[f"train_{k}"] = v
        
        val_bar = tqdm(total=len(val_dataloader), desc="Val")
        val_entry = loop(pipe, val_dataloader, device, args, None, criterion, val_bar, val=True)
        for k, v in val_entry.items():
            entry[f"val_{k}"] = v
        
        print(entry)

        history.append(entry)
        saveload1.save(history, is_ckpt=True)
        saveload2.save(history, is_ckpt=True)

    ## EVALUATION
    if args.test:
        # use dev
        test_builder = DSBuilder(
            triples_path=os.path.join(args.data_dir, "triples.json"),
            data_path=os.path.join(args.data_dir, "dev.jsonl"),
            n_reference_min=args.n_ref_min,
            n_reference_max=args.n_ref_max,
            stay_ratio_min=0.0,
            stay_ratio_max=0.0,
            random_state=args.seed,
            n_pick=1,
            items_path="./dev-items.jsonl",
            save_items=bool(args.save_items),
            load=bool(args.load_items)
        )

        test_ds = LMKBCDataset(
            test_builder,
            os.path.join(args.data_dir, "texts.txt"),
            os.path.join(args.data_dir, "entities.txt"),
            os.path.join(args.data_dir, "relations.txt"),
            os.path.join(args.data_dir, "entities_alias.jsonl"),
            n_tokens=args.n_token,
            tokenizer=tokenizer,
            texts_tensor_path=None,
            entities_tensor_path=None,
            relations_tensor_path=None,
            sentence_emb_mode=args.sentence_emb_mode,
            sentence_emb_index=args.sentence_emb_idx
        )

        test_ds.texts_attr = train_ds.texts_attr
        test_ds.entities_attr = train_ds.entities_attr
        test_ds.relations_attr = train_ds.relations_attr

        test_ds.prepare_eval(prompt_idx=0)

        test_collator = LMKBCCollator(test_ds, tokenizer, alias_idx=args.alias_idx)
        test_dataloader = DataLoader(test_ds, batch_size=args.bsize, shuffle=False, collate_fn=test_collator)

        test_bar = tqdm(total=len(test_dataloader), desc="Test")
        test_result = loop(pipe, test_dataloader, device, args, None, criterion, test_bar, val=True)

        with open(os.path.join(args.out, "test_metrics.json"), 'w') as fp:
            json.dump(test_result, fp)
from argparse import ArgumentParser
import os
import sys
sys.path.append("./src")

def init_args():
    parser = ArgumentParser()
    # DATA RELATED
    parser.add_argument("--train-data-dir", type=str, help="Data directory", default="./data/subgraph-gen/webnlg")
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
    parser.add_argument("--n-token-tensor", type=int, default=1)
    parser.add_argument("--n-token-gp", type=int, default=1)

    # MODEL
    parser.add_argument("--kgat", type=str, help="KGAT model path", required=True)
    parser.add_argument("--gp", type=str, help="Path to graph prefix checkpoint if exist")
    parser.add_argument("--bias", action="store_true")

    # TRAINING RELATED
    parser.add_argument("--freeze-kgat", action="store_true")
    parser.add_argument("--first-epoch", type=int, help="First epoch", default=5)
    parser.add_argument("--second-epoch", type=int, help="Second epoch", default=10)
    parser.add_argument("--bsize", type=int, help="Batch size", default=8)
    parser.add_argument("--lr", type=float, help="Learning rate", default=1e-3) # based on default adam, also mentioned in unimp paper
    parser.add_argument("--decay", type=float, help="Weight decay", default=0.0005) # based on unimp paper
    parser.add_argument("--weighted", action="store_true")
    parser.add_argument("--beam-augment", type=int, default=6)
    parser.add_argument("--beam-predict", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=32)

    # parser.add_argument("--train1", action="store_true") kalo skip pake first epoch 0
    # parser.add_argument("--train2", action="store_true") kalo skip pake second epoch 0
    
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

def loop(pipe, dataloader, device, args, optimizer, criterion, pbar, val=False): # train/val loop
    entry = {}
    start_time = time.time()
    if val:
        pipe.eval()
    else:
        pipe.train()

    total_loss = 0
    numel = 0

    for batch in dataloader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        labels = batch.pop("labels")
        weights = batch.pop("weights")
        batch["n_token"] = args.n_token_gp

        if val:
            with torch.no_grad():
                out = pipe.forward_lmkbc(**batch)
        else:
            out = pipe.forward_lmkbc(**batch)
        
        logits = out.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        weights = weights.repeat(shift_labels.shape[1], 1).transpose(0,1).reshape(-1)

        shift_logits = shift_logits.view(-1, pipe.language_model.config.vocab_size)
        shift_labels = shift_labels.view(-1)


        loss = criterion(shift_logits, shift_labels)

        if args.weighted:
            loss = loss * weights
        
        loss = loss[shift_labels != -100]
        mean_loss = loss.mean()

        total_loss = loss.sum().item() + total_loss
        numel += loss.numel()

        if not val:
            optimizer.zero_grad()
            mean_loss.backward()
            optimizer.step()
        pbar.update()
    
    end_time = time.time()

    entry["time"] = end_time - start_time
    entry["loss"] = total_loss / numel

    return entry

def generate(pipe, tokenizer, dataloader, device, args, pbar, augment=False):
    beam = args.beam_augment if augment else args.beam_predict

    pipe.eval()

    result = []

    for batch in dataloader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        
        batch["n_token"] = args.n_token_gp

        batch_size = batch["input_ids"].shape[0]
        
        out = pipe.generate_lmkbc(num_beams=beam, num_return_sequences=beam, max_new_tokens=args.max_new_tokens, 
                                  return_dict_in_generate=True, output_scores=True, **batch)
        
        sequence_ids = out.sequences

        transition_scores = pipe.language_model.compute_transition_scores(
            out.sequences, out.scores, out.beam_indices, normalize_logits=False
        )

        transition_scores = transition_scores.sum(-1) / (transition_scores != 0.0).sum(-1) # 0.0 is for padding token

        sequence_ids = sequence_ids.view(batch_size, beam, -1)
        transition_scores = transition_scores.view(batch_size, beam)

        text_out = []
        for s_id, ts in zip(sequence_ids, transition_scores):
            to = tokenizer.batch_decode(s_id, skip_special_tokens=True)
            # text_out.append(to)
            result.append({
                "text" : to,
                "score" : ts.tolist()
            })

        # preds.extend(text_out)
        
        pbar.update()
    
    return result

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
        items_path=os.path.join(args.data_dir, "train-items.jsonl"),
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
        items_path=os.path.join(args.data_dir, "dev-items.jsonl"),
        save_items=bool(args.save_items),
        load=bool(args.load_items)
    )

    texts_tensor_path = os.path.join(args.data_dir, f"texts.{args.lm.replace('/', '_')}.n_token={args.n_token_tensor}.index={args.sentence_emb_idx}.tensor")
    texts_tensor_path = texts_tensor_path if os.path.exists(texts_tensor_path) else os.path.join(args.data_dir, f"texts.{args.lm.replace('/', '_')}.n_token={args.n_token_tensor}.tensor")

    entities_tensor_path = os.path.join(args.data_dir, f"entities.{args.lm.replace('/', '_')}.n_token={args.n_token_tensor}.index={args.sentence_emb_idx}.tensor")
    entities_tensor_path = entities_tensor_path if os.path.exists(entities_tensor_path) else os.path.join(args.data_dir, f"entities.n_token={args.n_token_tensor}.tensor")

    relations_tensor_path = os.path.join(args.data_dir, f"relations.{args.lm.replace('/', '_')}.n_token={args.n_token_tensor}.index={args.sentence_emb_idx}.tensor")
    relations_tensor_path = relations_tensor_path if os.path.exists(relations_tensor_path) else os.path.join(args.data_dir, f"relations.n_token={args.n_token_tensor}.tensor")

    train_ds = LMKBCDataset(
        train_builder,
        os.path.join(args.data_dir, "texts.txt"),
        os.path.join(args.data_dir, "entities.txt"),
        os.path.join(args.data_dir, "relations.txt"),
        os.path.join(args.data_dir, "entities_alias.jsonl"),
        n_tokens=args.n_token_gp,
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
        n_tokens=args.n_token_gp,
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
    
    graph_prefix = GraphPrefix(in_channels=train_ds.texts_attr.shape[1], 
                               d_model=language_model.embed_dim, 
                               n_token=args.n_token_gp, bias=args.bias) if not args.gp else GraphPrefix.load(args.gp)

    pipe = Pipeline(kgat_model=kgat_model, graph_prefix=graph_prefix, language_model=language_model)
    
    ## TRAIN LOOP
    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pipe.kgat_model.to(device)
    pipe.graph_prefix.to(device)

    criterion = torch.nn.CrossEntropyLoss(reduction="none")
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
    if args.beam_augment > 0:
        train_ds.prepare_generate(prompt_idx=0)

        augment_collator = LMKBCCollator(train_ds, tokenizer, alias_idx=args.alias_idx, generate=True)
        augment_dataloader = DataLoader(train_ds, batch_size=args.bsize, shuffle=False, collate_fn=augment_collator)

        aug_bar = tqdm(total=len(augment_dataloader), desc="Augmentation")

        raw_predictions = generate(pipe, tokenizer, augment_dataloader, device, args, aug_bar, augment=True)


        train_ds.augment([el["text"] for el in raw_predictions])

        predictions = {
            "raw" : raw_predictions,
            "negative_objects" : train_ds.negative_objects
        }

        with open(os.path.join(args.out, "augment.json"), 'w') as fp:
            json.dump(predictions, fp)

    ## SECOND PHASE TRAIN
    train_ds.prepare_train(prompt_idx=args.prompt_idx)

    train_collator = LMKBCCollator(train_ds, tokenizer, alias_idx=args.alias_idx)
    val_collator = LMKBCCollator(val_ds, tokenizer, alias_idx=args.alias_idx)

    train_dataloader = DataLoader(train_ds, batch_size=args.bsize, shuffle=True, collate_fn=train_collator)
    val_dataloader = DataLoader(val_ds, batch_size=args.bsize, shuffle=False, collate_fn=val_collator)

    criterion = torch.nn.CrossEntropyLoss(reduction="none")
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
        # TEST
        test_builder = DSBuilder(
            triples_path=os.path.join(args.data_dir, "triples.json"),
            data_path=os.path.join(args.data_dir, "test.jsonl"),
            n_reference_min=args.n_ref_min,
            n_reference_max=args.n_ref_max,
            stay_ratio_min=0.0,
            stay_ratio_max=0.0,
            random_state=args.seed,
            n_pick=1,
            items_path=os.path.join(args.data_dir, "test-items.jsonl"),
            save_items=bool(args.save_items),
            load=bool(args.load_items)
        )

        test_ds = LMKBCDataset(
            test_builder,
            os.path.join(args.data_dir, "texts.txt"),
            os.path.join(args.data_dir, "entities.txt"),
            os.path.join(args.data_dir, "relations.txt"),
            os.path.join(args.data_dir, "entities_alias.jsonl"),
            n_tokens=args.n_token_gp,
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

        test_ds.prepare_generate(prompt_idx=0)

        test_collator = LMKBCCollator(test_ds, tokenizer, alias_idx=args.alias_idx, generate=True)
        test_dataloader = DataLoader(test_ds, batch_size=args.bsize, shuffle=False, collate_fn=test_collator)

        test_bar = tqdm(total=len(test_dataloader), desc="Predict test")
        
        predictions = generate(pipe, tokenizer, test_dataloader, device, args, test_bar, augment=False)
        
        with open(os.path.join(args.out, "preds-test.json"), 'w') as fp:
            json.dump(predictions, fp)

        # val

        # val_builder = DSBuilder(
        #     triples_path=os.path.join(args.data_dir, "triples.json"),
        #     data_path=os.path.join(args.data_dir, "dev.jsonl"),
        #     n_reference_min=args.n_ref_min,
        #     n_reference_max=args.n_ref_max,
        #     stay_ratio_min=0.0,
        #     stay_ratio_max=0.0,
        #     random_state=args.seed,
        #     n_pick=1,
        #     items_path=os.path.join(args.data_dir, "dev-items.jsonl"),
        #     save_items=bool(args.save_items),
        #     load=bool(args.load_items)
        # )

        # val_ds = LMKBCDataset(
        #     val_builder,
        #     os.path.join(args.data_dir, "texts.txt"),
        #     os.path.join(args.data_dir, "entities.txt"),
        #     os.path.join(args.data_dir, "relations.txt"),
        #     os.path.join(args.data_dir, "entities_alias.jsonl"),
        #     n_tokens=args.n_token_gp,
        #     tokenizer=tokenizer,
        #     texts_tensor_path=None,
        #     entities_tensor_path=None,
        #     relations_tensor_path=None,
        #     sentence_emb_mode=args.sentence_emb_mode,
        #     sentence_emb_index=args.sentence_emb_idx
        # )

        # val_ds.texts_attr = train_ds.texts_attr
        # val_ds.entities_attr = train_ds.entities_attr
        # val_ds.relations_attr = train_ds.relations_attr

        # val_ds.prepare_eval(prompt_idx=0)

        # val_collator = LMKBCCollator(val_ds, tokenizer, alias_idx=args.alias_idx)

        # val_dataloader = DataLoader(val_ds, batch_size=args.bsize, shuffle=False, collate_fn=val_collator)
        
        val_ds.prepare_generate(prompt_idx=0)

        val_collator = LMKBCCollator(val_ds, tokenizer, alias_idx=args.alias_idx, generate=True)
        val_dataloader = DataLoader(val_ds, batch_size=args.bsize, shuffle=False, collate_fn=val_collator)

        val_bar = tqdm(total=len(val_dataloader), desc="Predict val")
        
        predictions = generate(pipe, tokenizer, val_dataloader, device, args, val_bar, augment=False)
        
        with open(os.path.join(args.out, "preds-val.json"), 'w') as fp:
            json.dump(predictions, fp)
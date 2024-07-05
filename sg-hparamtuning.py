import json
import subprocess
import itertools
import shutil
import os

hyper_params = {
    "model_name_or_path" : ["meta-llama/Meta-Llama-3-8B"],
    # "checkpoint" : ["/raid/m13519061/.cache/huggingface/models--meta-llama--Meta-Llama-3-8B/snapshots/62bd457b6fe961a42a631306577e622c83876cb6"],
    "h_dim" : [512, 256, 128],
    "n_injector_head" : [8,4],
    "injector_dropout_p" : [0.1], 
    "encoder_dropout_p" : [0.1], 
    "n_encoder_head" : [8,4],
    "n_encoder_layers" : [6,4,2],
    "to_matrix" : ["diagonal", "outer_product"]
}

# CREATE COMBINATION USING ITERTOOLS
combinations = list(itertools.product(
    hyper_params["model_name_or_path"],
    hyper_params["h_dim"],
    hyper_params["n_injector_head"],
    hyper_params["injector_dropout_p"],
    hyper_params["encoder_dropout_p"],
    hyper_params["n_encoder_head"],
    hyper_params["n_encoder_layers"],
    hyper_params["to_matrix"]
))

# CREATE ./sg-paramtuning folder (exist_ok=True)
os.makedirs('./sg-paramtuning', exist_ok=True)

# FOR EVERY COMBINATION:
for i, combination in enumerate(combinations):
    if os.path.exists(f'./sg-paramtuning/{i}.json'):
        continue
    # Create a dictionary from the combination
    hparam_dict = {
        "model_name_or_path": combination[0],
        "h_dim" : combination[1],
        "n_injector_head": combination[2],
        "injector_dropout_p": combination[3],
        "encoder_dropout_p": combination[4],
        "n_encoder_head": combination[5],
        "n_encoder_layers": combination[6],
        "to_matrix": combination[7]
    }
    
    # WRITE THE DICTIONARY TO ./config/model/sg-hparam.json
    os.makedirs('./config/model', exist_ok=True)
    with open('./config/model/sg-hparam.json', 'w') as f:
        json.dump(hparam_dict, f, indent=4)
    
    # CALL accelerate launch sg-train.py --gpu 0,6,7 --n_data_train 1024 --n_data_val 1024 --n_data_test 1024 --bsize 4 --epoch 10 --data "./data/subgraph-gen/qagnn/csqa" --model "./config/model/sg-hparam.json" --best_metrics "sg_f1"
    subprocess.run([
        'accelerate', 'launch', 'sg-train.py', '--gpu', '0,1,2,3,4,5,6,7', '--n_data_train', '512', '--n_data_val', '256',
        '--n_data_test', '256', '--bsize', '2', '--epoch', '5', '--data', './data/subgraph-gen/qagnn/proc/csqa',
        '--model', './config/model/sg-hparam.json', '--best_metrics', 'sg_f1', '--lr', '0.00001'
    ])
    
    # CREATE ./sg-paramtuning folder/{i}.json WHERE i IS THE INDEX OF THE COMBINATION, ADD 3 KEYS : "evaluation_metrics", "history", "hparam"
    result_dict = {"hparam": hparam_dict}

    # "evaluation_metrics" VALUE IS FROM ./out/evaluation_metrics.json
    with open('./out/evaluation_metrics.json') as f:
        result_dict["evaluation_metrics"] = json.load(f)
    
    # "history" VALUE IS FROM ./out/history.json
    with open('./out/history.json') as f:
        result_dict["history"] = json.load(f)
    
    # Save the results in the sg-paramtuning folder
    with open(f'./sg-paramtuning/{i}.json', 'w') as f:
        json.dump(result_dict, f, indent=4)

    # REMOVE ./out
    shutil.rmtree('./out')

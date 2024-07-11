import json
import subprocess
import itertools
import shutil
import os

GPU = '0,1,2,3'
# COMBO_START = 50
ALPHA_START = 0.0
OUT_DIR = "./out"

# hyper_params = {
#     "model_name_or_path" : ["meta-llama/Meta-Llama-3-8B"],
#     # "checkpoint" : ["/raid/m13519061/.cache/huggingface/models--meta-llama--Meta-Llama-3-8B/snapshots/62bd457b6fe961a42a631306577e622c83876cb6"],
#     "h_dim" : [512, 256, 128],
#     "n_injector_head" : [8,4],
#     "injector_dropout_p" : [0.1], 
#     "encoder_dropout_p" : [0.1], 
#     "n_encoder_head" : [8,4],
#     "n_encoder_layers" : [6,4,2],
#     "to_matrix" : ["diagonal", "outer_product"]
# }

# # CREATE COMBINATION USING ITERTOOLS
# combinations = list(itertools.product(
#     hyper_params["model_name_or_path"],
#     hyper_params["h_dim"],
#     hyper_params["n_injector_head"],
#     hyper_params["injector_dropout_p"],
#     hyper_params["encoder_dropout_p"],
#     hyper_params["n_encoder_head"],
#     hyper_params["n_encoder_layers"],
#     hyper_params["to_matrix"]
# ))

# # CREATE ./sg-paramtuning folder (exist_ok=True)
# os.makedirs('./sg-paramtuning', exist_ok=True)

# # FOR EVERY COMBINATION:
# for i, combination in enumerate(combinations):
#     if os.path.exists(f'./sg-paramtuning/{i}.json') or i < COMBO_START:
#         continue
    # Create a dictionary from the combination

alphas = [el/10 for el in list(range(0,11,1))]

os.makedirs("./paramtuning/sg/iter4", exist_ok=True)

for a in alphas:

    if os.path.exists(f'./paramtuning/sg/iter4/{a}.json') and a < ALPHA_START:
        continue

    # CALL accelerate launch sg-train.py --gpu 0,6,7 --n_data_train 1024 --n_data_val 1024 --n_data_test 1024 --bsize 4 --epoch 10 --data "./data/subgraph-gen/qagnn/csqa" --model "./config/model/sg-hparam.json" --best_metrics "sg_f1"
    subprocess.run([
        'accelerate', 'launch', 'sg-train.py', '--gpu', GPU, '--n_data_train', '512', '--n_data_val', '512',
        '--n_data_test', '512', '--bsize', '2', '--epoch', '5', '--data', './data/subgraph-gen/qagnn/proc/csqa',
        '--model', './config/model/chosen.json', '--best_metrics', 'sg_f1', '--lr', '0.00001', '--split_size', '100', '--alpha', str(a), '--out', OUT_DIR
    ])

    result_dict = {}

    # "evaluation_metrics" VALUE IS FROM ./out/evaluation_metrics.json
    with open(f'{OUT_DIR}/evaluation_metrics.json') as f:
        result_dict["evaluation_metrics"] = json.load(f)

    # "history" VALUE IS FROM ./out/history.json
    with open(f'{OUT_DIR}/history.json') as f:
        result_dict["history"] = json.load(f)

    # Save the results in the sg-paramtuning folder
    with open(f'./paramtuning/sg/iter4/{a}.json', 'w') as f:
        json.dump(result_dict, f, indent=4)

    # REMOVE ./out
    shutil.rmtree(OUT_DIR)

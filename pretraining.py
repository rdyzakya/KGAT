import json
import subprocess
import shutil
import os

DATASETS = [
    'atomic/proc',
    'graph-writer/proc',
    'qagnn/proc/csqa',
    'qagnn/proc/obqa',
    'text2kg/proc',
    'webnlg/proc'
]

GPU = '0,1,2,3'
ALPHA = 0.9
OUT_DIR = "./pretraining-llama3"
SPLIT_SIZE = 50
N_PHASE = 80
DATA_PER_PHASE = 512
LR = 1e-5
BATCH_SIZE = 2
CONFIG = './config/model/llama3.json'
LOGGING_STEPS = 100

os.makedirs(os.path.join(OUT_DIR, "history"), exist_ok=True)

all_phases = [el for el in os.listdir(OUT_DIR) if el != "history"]
MAX_PHASE = -1
if len(all_phases) > 0:
    all_phases = [int(el) for el in all_phases]
    MAX_PHASE = max(all_phases)

for phase in range(N_PHASE):
    if os.path.exists(f'{OUT_DIR}/{phase}/checkpoint-0/model.pth') or phase < MAX_PHASE:
        continue
    ds = DATASETS[phase % len(DATASETS)]
    print(f"Training ./data/subgraph-gen/{ds}")
    start_index_train = (phase // len(DATASETS)) * DATA_PER_PHASE
    command = [
        'accelerate', 'launch', 'sg-train.py', '--gpu', GPU, '--n_data_train', str(DATA_PER_PHASE), '--bsize', str(BATCH_SIZE), '--epoch', '1',
        '--data', f'./data/subgraph-gen/{ds}', '--model', CONFIG, '--lr', str(LR), '--split_size', str(SPLIT_SIZE), 
        '--alpha', str(ALPHA), '--out', f'{OUT_DIR}/{phase}', '--start_index_train', str(start_index_train), '--no_test', '--no_val', '--dont_save',
        '--logging_steps', str(LOGGING_STEPS)
    ]

    if phase > 0:
        all_phases = [el for el in os.listdir(OUT_DIR) if el != "history"]
        all_phases = [int(el) for el in all_phases]
        MAX_PHASE = max(all_phases)
        command.extend([
            '--ckpt', f'{OUT_DIR}/{MAX_PHASE}/checkpoint-0/model.pth'
        ])
    subprocess.run(command)

    shutil.copy(f'{OUT_DIR}/{phase}/checkpoint-0/history.json', f'{OUT_DIR}/history/{phase}.json')
    shutil.rmtree(f'{OUT_DIR}/{phase}')
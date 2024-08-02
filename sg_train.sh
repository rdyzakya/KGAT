python sg_train.py --data-dir ./data/subgraph-gen/webnlg \
                    --n-ref-min 10 \
                    --n-ref-max 50 \
                    --stay-ratio-min 1.0 \
                    --stay-ratio-max 1.0 \
                    --save-items \
                    --sentence-emb-mode baseline \
                    --lm openai-community/gpt2 \
                    --sentence-emb-idx 11 \
                    --alias-idx 0 \
                    --hidden 1536 \
                    --layer 2 \
                    --head 1 \
                    --epoch 5 \
                    --bsize 8 \
                    --lr 0.0001 \
                    --decay 0.5 \
                    --estop \
                    --estop-patience 3 \
                    --estop-delta 0.05 \
                    --best-metrics link_f1 \
                    --load-best \
                    --max-ckpt 3 \
                    --test \
                    --seed 42
                    # --super-set
                    # --all
                    # --gpu 0,1,2,3
python sg_train.py --data-dir ./data/subgraph-gen/graphtextqa \
                    --n-ref-min 10 \
                    --n-ref-max 50 \
                    --stay-ratio-min 1.0 \
                    --stay-ratio-max 1.0 \
                    --save-items \
                    --sentence-emb-mode eol \
                    --lm openai-community/gpt2 \
                    --sentence-emb-idx 11 \
                    --alias-idx 0 \
                    --d-ff 768 \
                    --head 1 \
                    --n-block 2 \
                    --bias \
                    --relation \
                    --dropout 0.3 \
                    --epoch 3 \
                    --bsize 8 \
                    --lr 0.001 \
                    --decay 0.0005 \
                    --estop \
                    --estop-patience 3 \
                    --estop-delta 0.05 \
                    --best-metrics f1 \
                    --load-best \
                    --max-ckpt 3 \
                    --test \
                    --seed 42 \
                    --load-items
                    # --beta \
                    # --super-set
                    # --all
                    # --gpu 0,1,2,3
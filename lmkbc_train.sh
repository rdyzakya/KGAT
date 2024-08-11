python lmkbc_train.py --data-dir ./data/lmkbc/2022 \
                    --n-ref-min 50 \
                    --n-ref-max 50 \
                    --stay-ratio-min 0.0 \
                    --stay-ratio-max 0.0 \
                    --save-items \
                    --sentence-emb-mode eol \
                    --lm openai-community/gpt2 \
                    --sentence-emb-idx 11 \
                    --alias-idx 0 \
                    --prompt-idx 0 \
                    --n-token 1 \
                    --kgat ./out/model.pth \
                    --freeze-kgat \
                    --first-epoch 5 \
                    --second-epoch 5 \
                    --bsize 8 \
                    --lr 0.00001 \
                    --decay 0.0005 \
                    --weighted \
                    --beam 3 \
                    --estop \
                    --estop-patience 3 \
                    --estop-delta 0.05 \
                    --best-metrics f1 \
                    --load-best \
                    --max-ckpt 3 \
                    --test \
                    --seed 42 \
                    --load-items \
                    --out ./lmkbc-test
                    # --relation \
                    # --beta \
                    # --super-set
                    # --all
                    # --gpu 0,1,2,3
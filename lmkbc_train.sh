python lmkbc_train.py --data-dir ./data/lmkbc/2022 \
                    --n-ref-min 50 \
                    --n-ref-max 50 \
                    --stay-ratio-min 0.0 \
                    --stay-ratio-max 0.0 \
                    --save-items \
                    --sentence-emb-mode ke \
                    --lm google/gemma-2-27b \
                    --sentence-emb-idx 45 \
                    --alias-idx 0 \
                    --prompt-idx 0 \
                    --n-token-gp 4 \
                    --n-token-tensor 1 \
                    --kgat ./out/model.pth \
                    --bias \
                    --first-epoch 5 \
                    --second-epoch 5 \
                    --bsize 8 \
                    --lr 1e-5 \
                    --decay 5e-4 \
                    --weighted \
                    --max-new-tokens 32 \
                    --estop \
                    --estop-patience 3 \
                    --estop-delta 0.05 \
                    --best-metrics loss \
                    --load-best \
                    --max-ckpt 5 \
                    --test \
                    --seed 42 \
                    --load-items \
                    --beam-augment 6 \
                    --beam-predict 6 \
                    --gpu 0,1,2,3,4 \
                    --out ./lmkbc-gemma
                    # --freeze-kgat \
                    # --gp ./out/model.pth
                    # --relation \
                    # --beta \
                    # --super-set
                    # --all
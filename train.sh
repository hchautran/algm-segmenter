python -m segm.train  --log-dir runs/vit_small_layers_1_5_T_0.88/ \
                      --dataset ade20k \
                      --backbone vit_small_patch16_384 \
                      --decoder mask_transformer \
                      --patch-type algm \
                      --selected-layers 1 5 \
                      --merging-window-size 2 2 \
                      --threshold 0.88 
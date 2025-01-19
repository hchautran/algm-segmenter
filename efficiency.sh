# # Im/sec
# python -m segm.speedtest --model-dir /media/caduser/MyBook/chau/algm-segmenter/runs/vit_ti_16 \
#         --dataset ade20k \
#         --batch-size 1 \
#         --patch-type pure

# python -m segm.speedtest --model-dir /media/caduser/MyBook/chau/algm-segmenter/runs/vit_ti_16 \
#         --dataset ade20k \
#         --batch-size 1 \
#         --patch-type algm \
#         --selected-layers 1 5 \
#         --merging-window-size 2 2 \
#         --threshold 0.75

# python -m segm.speedtest --model-dir /media/caduser/MyBook/chau/algm-segmenter/runs/vit_ti_16 \
#         --dataset ade20k \
#         --batch-size 1 \
#         --patch-type pitome \
#         --selected-layers 1 5 \
#         --merging-window-size 2 2 \
#         --threshold 0.90

# GFLOPs





# CUDA_VISIBLE_DEVICES=1 python -m segm.flops --model-dir /media/caduser/MyBook/chau/algm-segmenter/runs/vit_ti_16 \
#           --dataset ade20k \
#           --batch-size 1 \
#           --patch-type algm \
#           --selected-layers 1 5 \
#           --merging-window-size 2 2 \
#           --threshold 0.88



CUDA_VISIBLE_DEVICES=1 python -m segm.flops --model-dir /media/caduser/MyBook/chau/algm-segmenter/runs/vit_ti_16 \
          --dataset ade20k \
          --batch-size 1 \
          --patch-type pitome\
          --selected-layers 1 5 \
          --merging-window-size 2 2 \
          --threshold 0.925


# GFLOPs
python -m segm.flops --model-dir /media/caduser/MyBook/chau/algm-segmenter/runs/vit_ti_16 \
          --dataset ade20k \
          --batch-size 1 \
          --patch-type pure 
        #   --selected-layers 1 5 \
        #   --merging-window-size 2 2 \
        #   --threshold 0.88


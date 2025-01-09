# # single-scale baseline evaluation:
# python -m segm.eval.miou runs/vit_small_layers_1_5_T_0.88/checkpoint.pth \
#           ade20k \
#           --singlescale \
#           --patch-type pure 

# Explanation:
# --singlescale: Evaluates the model using a single scale of input images.
# --patch-type pure: Uses the standard patch processing without any modifications.

# single-scale baseline + ALGM evaluation:
python -m segm.eval.miou runs/vit_small_layers_1_5_T_0.88/checkpoint.pth \
          ade20k \
          --singlescale \
          --patch-type algm \
          --selected-layers 1 5  \
          --merging-window-size 2 2 \
          --threshold 0.88

# Explanation:
# --patch-type algm: Applies the ALGM patch type.
# --selected-layers 1 5: Specifies which layers of the network to apply ALGM. In this case, layers 1 and 5.
# --merging-window-size 2 2: Sets the size of the merging window for the ALGM algorithm, here it is 2x2.
# --threshold 0.90: Sets the confidence threshold for merging patches in ALGM, where 0.90 stands for 90% confidence.

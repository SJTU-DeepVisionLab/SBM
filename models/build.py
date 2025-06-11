# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Demystify Mamba in Vision: A Linear Attention Perspective
# Modified by Dongchen Han
# -----------------------------------------------------------------------

from .sbm import SBM

def build_model(config):
    model_type = config.MODEL.TYPE

    if model_type == 'sbm':
        model = SBM(img_size=config.DATA.IMG_SIZE,
                     patch_size=config.MODEL.SBM.PATCH_SIZE,
                     in_chans=config.MODEL.SBM.IN_CHANS,
                     num_classes=config.MODEL.NUM_CLASSES,
                     embed_dim=config.MODEL.SBM.EMBED_DIM,
                     depths=config.MODEL.SBM.DEPTHS,
                     mlp_ratio=config.MODEL.SBM.MLP_RATIO,
                     drop_path_rate=config.MODEL.DROP_PATH_RATE,
                    #  ape=config.MODEL.SBM.APE,
                     use_checkpoint=config.TRAIN.USE_CHECKPOINT)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model

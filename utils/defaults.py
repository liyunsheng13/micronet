import os
from yacs.config import CfgNode as CN


# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.DEBUG = False # add debug flag

# If the WEIGHT starts with a catalog://, like :R-50, the code will look for
# the path in paths_catalog. Else, it will use it as the specified absolute
# path
_C.MODEL.WEIGHT = ""

_C.MODEL.ACTIVATION = CN()
_C.MODEL.ACTIVATION.MODULE = "MaxLUConv" # old for mbnetm2 "MaxLUConv"
_C.MODEL.ACTIVATION.ACT_MAX = 1.0
_C.MODEL.ACTIVATION.LAST_SE_OUP = False #use se-oup for the last 1x1 conv
_C.MODEL.ACTIVATION.LINEARSE_BIAS = True
_C.MODEL.ACTIVATION.INIT_A_BLOCK3 = [1.0, 0.0]
_C.MODEL.ACTIVATION.INIT_A = [1.0, 0.0]
_C.MODEL.ACTIVATION.INIT_B = [0.0, 0.0]
_C.MODEL.ACTIVATION.REDUCTION = 4
_C.MODEL.ACTIVATION.FC = False
_C.MODEL.ACTIVATION.ACT = 'relu'

_C.MODEL.MICRONETS = CN()
_C.MODEL.MICRONETS.NET_CONFIG = "d12_3322_192_k5"
_C.MODEL.MICRONETS.STEM_CH = 16
_C.MODEL.MICRONETS.STEM_DILATION = 1
_C.MODEL.MICRONETS.STEM_GROUPS = [4, 8]
_C.MODEL.MICRONETS.STEM_MODE = "default" # defaut/max2
_C.MODEL.MICRONETS.BLOCK = "MicroBlock1"
_C.MODEL.MICRONETS.POINTWISE = 'group' #fft/1x1/shuffle
_C.MODEL.MICRONETS.DEPTHSEP = True # YUNSHENG ADD FOR MUTUAL LEARNING
_C.MODEL.MICRONETS.SHUFFLE = False
_C.MODEL.MICRONETS.OUT_CH = 1024
_C.MODEL.MICRONETS.DROPOUT = 0.0


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "."

_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")

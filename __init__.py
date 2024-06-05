from .node import *
from .install import *

NODE_CLASS_MAPPINGS = {
    "Apply OmDet": ApplyOmDet,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Apply OmDet": "Apply OmDet",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

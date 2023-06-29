from .resnet3 import resnet18 as resnet18_3
from .resnet3 import resnet50 as resnet50_3
from .resnet3 import resnet101 as resnet101_3
from .resnet7 import resnet18 as resnet18_7
from .resnet7 import resnet50 as resnet50_7
from .resnet7 import resnet101 as resnet101_7

from .builder import build_backbone

__all__ = ['resnet18_3', 'resnet50_3', 'resnet101_3', 'resnet18_7', 'resnet50_7', 'resnet101_7']

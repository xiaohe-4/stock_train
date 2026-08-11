"""赛事目录约定的特征工程接口。"""

from pathlib import Path
import sys

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
from utils import engineer_features_39, engineer_features_158plus39

__all__ = ['engineer_features_39', 'engineer_features_158plus39']

"""赛事目录约定的特征工程接口。

实际实现位于 code/utils.py；此模块提供稳定入口，避免复制两套特征计算逻辑。
"""

from utils import engineer_features_39, engineer_features_158plus39

__all__ = ['engineer_features_39', 'engineer_features_158plus39']

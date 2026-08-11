"""兼容赛事目录约定的训练入口。"""

from pathlib import Path
import runpy
import sys

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
runpy.run_path(str(CODE_DIR / 'train.py'), run_name='__main__')

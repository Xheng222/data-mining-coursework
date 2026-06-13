"""合成数据生成与 CleanML 真实数据集加载。

公开入口：
  - ``generate_dataset`` —— 合成数据（含缺陷注入），见 ``src/datagen/generate.py``。
  - ``load_cleanml``     —— CleanML 真实脏/净配对数据集加载，见 ``src/datagen/cleanml.py``。
"""

from src.datagen.cleanml import load_cleanml  # noqa: F401
from src.datagen.generate import generate_dataset  # noqa: F401

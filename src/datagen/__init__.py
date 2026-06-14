"""合成数据生成、CleanML 加载与真实基座加载。

公开入口：
  - ``generate_dataset``  —— 合成数据（含缺陷注入），见 ``src/datagen/generate.py``。
  - ``load_cleanml``      —— CleanML 真实脏/净配对数据集，见 ``src/datagen/cleanml.py``。
  - ``load_cleanml_clean``—— 仅加载 CleanML 的干净版本 DataFrame（用于受控注入）。
  - ``load_real_dataset`` —— 真实数据基座（UCI/sklearn），见 ``src/datagen/real_base.py``。
"""

from src.datagen.cleanml import load_cleanml, load_cleanml_clean  # noqa: F401
from src.datagen.generate import generate_dataset  # noqa: F401
from src.datagen.real_base import load_real_dataset  # noqa: F401

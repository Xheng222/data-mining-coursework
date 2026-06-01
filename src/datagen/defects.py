"""六类缺陷注入函数。feature/datagen 负责实现。

每个 inject_* 接收 DataFrame，返回 (注入后的 DataFrame, ground_truth 片段)。
ground_truth 片段格式见 src/contracts.py 的 DefectReport schema。
"""

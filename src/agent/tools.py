"""Executor 可调用的数据操作工具。feature/agent 负责实现。

例如：profile（缺失/重复/相关性统计）、impute、drop_duplicates、
drop_columns、train_xgb 等。工具应是确定性的纯 pandas/sklearn 操作。
"""

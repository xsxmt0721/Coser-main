# Core/utils.py

import json
from typing import List, Dict, Any
from datetime import datetime


def preprocess_dataset(raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    对原始数据集进行预处理，使其符合精简的微调范式：
    {
      "instruction": str,
      "input": str,
      "output": str
    }
    同时，强制字段类型为字符串，并移除冗余或空的 history 字段。
    """
    cleaned_data = []

    # 定义标准字段及其预期的默认值和类型
    standard_fields = {
        "instruction": ("", str),
        "input": ("", str),
        "output": ("", str),
    }

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🧹 开始对 {len(raw_data)} 条数据进行预处理（移除 history 字段）...")

    for i, item in enumerate(raw_data):
        cleaned_item = {}
        valid = True

        # 遍历标准字段，确保其存在且类型正确
        for field, (default_value, expected_type) in standard_fields.items():
            value = item.get(field)

            # --- 核心清洗逻辑：确保是字符串 ---
            if value is None:
                cleaned_item[field] = default_value
            else:
                try:
                    # 确保 instruction/input/output 是字符串，并去除首尾空格
                    cleaned_item[field] = str(value).strip()
                except Exception:
                    # 类型转换失败，使用默认值
                    cleaned_item[field] = default_value

            # --- 校验：确保关键字段不为空 ---
            if field in ['instruction', 'output'] and not cleaned_item[field]:
                valid = False
                break  # 跳过当前条目

        # 如果数据有效且不为空，则添加到清洗列表中
        if valid and cleaned_item['instruction'] and cleaned_item['output']:
            cleaned_data.append(cleaned_item)

    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] 预处理完成。原始 {len(raw_data)} 条，清洗后 {len(cleaned_data)} 条。")
    return cleaned_data
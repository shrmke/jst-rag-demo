#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import pandas as pd
import sys
from pathlib import Path

def jsonl_to_xlsx(input_jsonl: str, output_xlsx: str) -> None:
    """将jsonl文件转换为xlsx文件"""
    
    print(f"读取jsonl文件: {input_jsonl}")
    
    # 读取jsonl文件
    data_list = []
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                data_list.append(data)
            except json.JSONDecodeError as e:
                print(f"警告: 第{line_num}行JSON解析失败: {e}")
                continue
    
    if not data_list:
        print("错误: jsonl文件中没有有效数据")
        return
    
    print(f"共读取 {len(data_list)} 条记录")
    
    # 转换为DataFrame
    df = pd.DataFrame(data_list)
    
    # 保存为xlsx
    print(f"保存为xlsx文件: {output_xlsx}")
    df.to_excel(output_xlsx, index=False, engine='openpyxl')
    
    print(f"转换完成！输出文件: {output_xlsx}")
    print(f"列名: {', '.join(df.columns.tolist())}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # 默认转换benchmark_results_v1.jsonl
        input_file = "/home/wangyaqi/jst/测试文件/benchmark_results_v2.jsonl"
        output_file = "/home/wangyaqi/jst/测试文件/benchmark_results_v2.xlsx"
    elif len(sys.argv) == 2:
        # 只提供输入文件，输出文件名自动生成
        input_file = sys.argv[1]
        output_path = Path(input_file)
        output_file = str(output_path.with_suffix('.xlsx'))
    else:
        # 提供输入和输出文件
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    
    jsonl_to_xlsx(input_file, output_file)
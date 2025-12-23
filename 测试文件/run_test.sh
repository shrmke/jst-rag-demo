#!/bin/bash

# 启动虚拟环境并运行基准测试
cd /home/wangyaqi
source venv/bin/activate

# 运行测试
cd /Users/wangyaqi/Documents/cursor_project/jst-rag-demo/jst-rag-demo/测试文件
python3 test_benchmark.py

echo "测试完成！结果文件：benchmark_results.xlsx"

# 基准测试脚本

## 文件说明

- `test_benchmark.py`: 主要测试脚本
- `run_test.sh`: 运行脚本（自动激活虚拟环境）
- `金盘benchmark测试.xlsx`: 测试数据（第一列为问题）
- `benchmark_results.xlsx`: 测试结果输出

## 使用方法

### 方法1：直接运行Python脚本
```bash
cd /home/wangyaqi
source venv/bin/activate
cd jst/测试文件
python3 test_benchmark.py
```

### 方法2：使用运行脚本
```bash
cd /Users/wangyaqi/Documents/cursor_project/jst-rag-demo/jst-rag-demo/测试文件
./run_test.sh
```

## 测试参数

- **BM25+Embedding预选**: pre_topk=30
- **重排返回**: rerank_topk=10
- **每子库向量检索**: topK=50
- **每子库BM25检索**: topK=50
- **邻近chunk半径**: 1
- **用于回答的TopK**: 30
- **BM25权重α**: 0.5
- **命中表格返回整表**: 开启
- **Multi-Query/HyDE**: 关闭

## 输出格式

`benchmark_results.xlsx` 包含以下列：
1. **问题**: 原始测试问题
2. **大模型回答**: AI生成的回答
3. **识别出的年份**: 意图识别出的年份列表
4. **识别出的财报/公告**: 识别出的文档类型（report/notice/unknown）

## 注意事项

- 确保已安装所需Python包：pandas, openpyxl
- 确保虚拟环境已激活
- 测试需要网络连接（调用Qwen API）
- 测试可能需要较长时间，取决于问题数量


测试参数
v1:
测试参数配置：
✅ BM25+Embedding预选: pre_topk=30
✅ 重排返回: rerank_topk=10
✅ 每子库向量检索: topK=50
✅ 每子库BM25检索: topK=50
✅ 邻近chunk半径: 1
✅ 用于回答的TopK: 30
✅ BM25权重α: 0.5
✅ 命中表格返回整表: 开启
✅ Multi-Query/HyDE: 关闭

v2:
测试参数配置：
✅ BM25+Embedding预选: pre_topk=30
✅ 重排返回: rerank_topk=30
✅ 每子库向量检索: topK=50
✅ 每子库BM25检索: topK=50
✅ 邻近chunk半径: 1
✅ 用于回答的TopK: 30
✅ BM25权重α: 0.5
✅ 命中表格返回整表: 开启
✅ Multi-Query/HyDE: 关闭

v3:
相比v2去掉了query中的“\n”
测试参数配置：
✅ BM25+Embedding预选: pre_topk=30
✅ 重排返回: rerank_topk=30
✅ 每子库向量检索: topK=50
✅ 每子库BM25检索: topK=50
✅ 邻近chunk半径: 1
✅ 用于回答的TopK: 30
✅ BM25权重α: 0.5
✅ 命中表格返回整表: 开启
✅ Multi-Query/HyDE: 关闭
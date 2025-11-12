## 索引构建与检索（FAISS + BM25）

环境变量（沿用 RAG-Challenge-2/Qwen 设置）
- `QWEN_API_KEY`、`QWEN_API_BASE`（默认 `https://dashscope.aliyuncs.com/compatible-mode/v1`）
- `QWEN_EMBED_MODEL`（默认 `qwen-embedding-v4`）

安装（如缺依赖）
```bash
pip install faiss-cpu rank-bm25 numpy
```

构建索引
- 实验组（不 strip HTML）：
```bash
python3 /home/wangyaqi/jst/build_indexes.py \
  --chunks-dir /home/wangyaqi/jst/chunks_exp \
  --out-faiss /home/wangyaqi/jst/indexes/faiss_exp \
  --out-bm25  /home/wangyaqi/jst/indexes/bm25_exp
```
- 对照组（strip HTML，用于嵌入；回显仍用原始 HTML）：
```bash
python3 /home/wangyaqi/jst/build_indexes.py \
  --chunks-dir /home/wangyaqi/jst/chunks_ctrl \
  --out-faiss /home/wangyaqi/jst/indexes/faiss_ctrl \
  --out-bm25  /home/wangyaqi/jst/indexes/bm25_ctrl \
  --strip-html
```

检索
```bash
# 实验组
python3 /home/wangyaqi/jst/search.py \
  --variant exp \
  --faiss-dir /home/wangyaqi/jst/indexes/faiss_exp \
  --bm25-dir  /home/wangyaqi/jst/indexes/bm25_exp \
  --query "募投项目 调整后金额" --topk 10

# 对照组（支持混合检索，alpha 越大 BM25 权重越高）
python3 /home/wangyaqi/jst/search.py \
  --variant ctrl \
  --faiss-dir /home/wangyaqi/jst/indexes/faiss_ctrl \
  --bm25-dir  /home/wangyaqi/jst/indexes/bm25_ctrl \
  --query "可转债 转股价格" --topk 10 --alpha 0.3
```

输出与目录结构
- FAISS 目录：`faiss.index` + `meta.jsonl`（与向量一一对应）
- BM25 目录：`bm25.pkl` + `meta.jsonl`
- `meta.jsonl` 每行包含：`id/doc_id/page/type/order/embed_text/raw_content/table/meta/source_file`



启动虚拟环境
cd /home/wangyaqi
source venv/bin/activate

原pdf：861
解析后：853
原因暂未查看

rag框架
0.意图识别和问题拆解
尝试：
分为财报类和公告类
用大模型识别，如果意图想问金盘科技某年或某时间段具体的项目，则走财报
其他走公告类

问题拆解：复杂问题拆解成多个独立问题独立检索


1.ocr
模型：新mineru
输入：金盘上市公告
输出：金盘上市公告_mineru解析
type类型：text,table,page_number,image,list,equation,header
需要：text,table,list,equation
问题：原pdf：861；解析后：853；原因暂未查看

2.table2text
（暂未对跨页表格和跨页单元格做特殊处理）
实验组：
公告类：
对content_list.json中的table类型做处理，脚本检测是否包含合并单元格，如果不包含，则视为简单表格，走脚本进行文本化（有些表没有表头，会引起转文本错误）；
如果包含，则走大模型进行文本化，将表格信息处理为行级文本
输入：金盘上市公告_mineru解析
输出：金盘上市公告_table2text
命令：python /home/wangyaqi/jst/textify_tables.py "/home/wangyaqi/jst/金盘公告_mineru解析"
两三天

财报类：
对公告类类似，但是在表格中用大模型注入时间
输入：金盘财报_mineru解析
输出：金盘财报_table2text
命令：python /home/wangyaqi/jst/textify_tables_llm.py "/home/wangyaqi/jst/金盘财报_mineru解析"
对比组：不做任何处理

3.chunking
实验组：
对text类型，进行固定长度切分；去掉header类型；对table类型，按table2text进行行级切分；
财报：python /home/wangyaqi/jst/chunk_content.py "/home/wangyaqi/jst/金盘财报_table2text" --mode exp --out-dir "/home/wangyaqi/jst/金盘财报_chunks"
公告：python /home/wangyaqi/jst/chunk_content.py "/home/wangyaqi/jst/金盘上市 公告_table2text" --mode exp --out-dir "/home/wangyaqi/jst/金盘上市公告_chunks"
对比组：
进行固定长度切分，同时保留元素完整（要考虑到过长的table），如果加上next unit则大于maxlen，那么舍弃next unit，保证每个chunk都小于等于maxlen
（不要对比组了）


4.embedding
模型：找现在比较好的模型

5.retrieve
问题：如何分类，如何意图识别？要根据公司名分，但一个公司内按什么分？按年份分我觉得不太合适，2018年的数据极有可能也会出现在2019年的报表上
vector retrieve
bm25
reranking

命令：
python3 /home/wangyaqi/jst/search.py   --kb 财报   --query "2021年第一季度公司总资产是多少？"   --pre-topk 30   --rerank-topk 10   --faiss-per-index 10   --bm25-per-index 50   --alpha 0.5   --neighbor-radius 1   --return-table-full

6.generate
检索到表格，则返回原整个表格及其上下文chunk
检索到文本，则同时返回其上下chunk

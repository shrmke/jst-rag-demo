#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import json
import argparse
from typing import List, Dict, Any

# 添加项目根目录到路径
sys.path.insert(0, '/home/wangyaqi/jst')

from search import route_and_search, build_answer_messages, call_chat_completion, QWEN_CHAT_MODEL

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='运行基准测试或单条查询')
    parser.add_argument('--query', type=str, help='单条查询字符串，如果提供则执行单条查询而不是批量测试')
    parser.add_argument('--input', type=str, default="/home/wangyaqi/jst/测试文件/金盘benchmark测试.xlsx",
                       help='批量测试输入Excel文件路径')
    parser.add_argument('--output', type=str, default="/home/wangyaqi/jst/测试文件/benchmark_results_v2.jsonl",
                       help='批量测试输出JSONL文件路径')
    return parser.parse_args()

def run_single_query(query: str) -> None:
    """运行单条查询并直接输出结果"""
    print(f"处理查询: {query}")
    print("-" * 50)

    try:
        # 调用检索和回答生成
        search_response = route_and_search(
            query=query,
            topk=30,  # 用于回答的TopK条数=30
            alpha=0.5,  # BM25权重α
            pre_topk=30,  # BM25+Embedding预选
            faiss_per_index=50,  # 每子库向量检索topK
            bm25_per_index=50,  # 每子库BM25检索topK
            rerank_topk=30,  # 重排返回topK
            rerank_instruct="Given a web search query, retrieve relevant passages that answer the query.",
            neighbor_radius=1,  # 返回上下相邻chunk半径
            return_table_full=True,  # 命中表格返回整表
            output_cap=200,
            prefer_year=True,
            doc_type_override=None,  # 自动识别
        )

        # 提取意图信息
        intent_dict = search_response.get("intent", {})
        retrieved_chunks = search_response.get("results", [])

        # 输出意图识别结果
        detected_years = intent_dict.get("years", [])
        years_display = ", ".join(map(str, detected_years)) if detected_years else "无"
        detected_doc_type = intent_dict.get("doc_type", "")
        doc_type_display = detected_doc_type if detected_doc_type else "unknown"

        print(f"识别出的年份: {years_display}")
        print(f"识别出的文档类型: {doc_type_display}")
        print(f"检索到 {len(retrieved_chunks)} 个相关片段")
        print()

        # 生成回答
        if retrieved_chunks:
            contexts_for_answer = retrieved_chunks[:30]  # 用于回答的TopK条数=30
            try:
                answer_messages = build_answer_messages(
                    query=query,
                    contexts=contexts_for_answer,
                    system_prompt="你是严谨的中文金融助理。基于给定检索片段回答，若无法确定，请明确说明。引用用 [编号] 标注。",
                    per_chunk_limit=1200,  # 每条上下文最大字符数
                    include_full_table=True,
                )
                model_answer = call_chat_completion(
                    messages=answer_messages,
                    model=QWEN_CHAT_MODEL,
                    temperature=0.1,  # 回答temperature
                    max_tokens=512,   # 回答max_tokens
                )
                print("大模型回答:")
                print(model_answer)
            except Exception as e:
                print(f"回答生成失败: {str(e)}")
        else:
            print("未找到相关检索结果")

    except Exception as e:
        print(f"查询处理失败: {str(e)}")

def run_benchmark_test(input_xlsx: str, output_jsonl: str) -> None:
    """运行基准测试"""

    # 读取测试问题
    print("读取测试数据...")
    df = pd.read_excel(input_xlsx)
    questions = df.iloc[:, 0].tolist()  # 第一列是问题

    processed_count = 0  # 已处理数量计数

    # 打开输出文件（覆盖模式，如果需要追加可以改为'a'）
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for i, question in enumerate(questions, 1):
            question = question.strip()  # 去掉首尾空白字符，包括换行符
            print(f"处理问题 {i}/{len(questions)}: {question[:50]}...")

            try:
                # 调用检索和回答生成
                search_response = route_and_search(
                    query=question,
                    topk=30,  # 用于回答的TopK条数=30
                    alpha=0.5,  # BM25权重α
                    pre_topk=30,  # BM25+Embedding预选
                    faiss_per_index=50,  # 每子库向量检索topK
                    bm25_per_index=50,  # 每子库BM25检索topK
                    rerank_topk=30,  # 重排返回topK
                    rerank_instruct="Given a web search query, retrieve relevant passages that answer the query.",
                    neighbor_radius=1,  # 返回上下相邻chunk半径
                    return_table_full=True,  # 命中表格返回整表
                    output_cap=200,
                    prefer_year=True,
                    doc_type_override=None,  # 自动识别
                )

                # 提取意图信息
                intent_dict = search_response.get("intent", {})
                # 提取检索结果列表（使用明确的变量名避免冲突）
                retrieved_chunks = search_response.get("results", [])

                # 生成回答
                model_answer = ""
                if retrieved_chunks:
                    contexts_for_answer = retrieved_chunks[:30]  # 用于回答的TopK条数=30
                    try:
                        answer_messages = build_answer_messages(
                            query=question,
                            contexts=contexts_for_answer,
                            system_prompt="你是严谨的中文金融助理。基于给定检索片段回答，若无法确定，请明确说明。引用用 [编号] 标注。",
                            per_chunk_limit=1200,  # 每条上下文最大字符数
                            include_full_table=True,
                        )
                        model_answer = call_chat_completion(
                            messages=answer_messages,
                            model=QWEN_CHAT_MODEL,
                            temperature=0.1,  # 回答temperature
                            max_tokens=512,   # 回答max_tokens
                        )
                    except Exception as e:
                        model_answer = f"回答生成失败: {str(e)}"

                # 获取年份
                detected_years = intent_dict.get("years", [])
                years_display = ", ".join(map(str, detected_years)) if detected_years else ""

                # 获取文档类型
                detected_doc_type = intent_dict.get("doc_type", "")
                doc_type_display = detected_doc_type if detected_doc_type else "unknown"

                # 构建单条测试结果
                single_result = {
                    "问题": question,
                    "大模型回答": model_answer,
                    "识别出的年份": years_display,
                    "识别出的财报/公告": doc_type_display
                }

                # 立即写入结果到文件
                json_line = json.dumps(single_result, ensure_ascii=False)
                f.write(json_line + '\n')
                f.flush()  # 确保立即写入磁盘
                processed_count += 1

                print(f"✓ 已保存结果 {processed_count}/{len(questions)}")

            except Exception as e:
                print(f"处理问题失败: {e}")
                error_result = {
                    "问题": question,
                    "大模型回答": f"处理失败: {str(e)}",
                    "识别出的年份": "",
                    "识别出的财报/公告": ""
                }
                # 立即写入错误结果到文件
                json_line = json.dumps(error_result, ensure_ascii=False)
                f.write(json_line + '\n')
                f.flush()  # 确保立即写入磁盘
                processed_count += 1

                print(f"✗ 已保存错误结果 {processed_count}/{len(questions)}")

    print(f"\n测试完成，结果保存到: {output_jsonl}")
    print(f"共处理 {processed_count} 条结果")

if __name__ == "__main__":
    args = parse_arguments()

    if args.query:
        # 执行单条查询
        run_single_query(args.query)
    else:
        # 执行批量测试
        run_benchmark_test(args.input, args.output)

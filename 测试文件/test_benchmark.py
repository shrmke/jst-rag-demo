#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import json
import argparse
import re
from datetime import datetime
from typing import List, Dict, Any

# 添加项目根目录到路径
sys.path.insert(0, '/Users/wangyaqi/Documents/cursor_project/jst-rag-demo/jst-rag-demo')

from search import route_and_search, build_answer_messages, call_chat_completion, QWEN_CHAT_MODEL

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='运行基准测试或单条查询')
    parser.add_argument('--query', type=str, help='单条查询字符串，如果提供则执行单条查询而不是批量测试')
    parser.add_argument('--input', type=str, default="/Users/wangyaqi/Documents/cursor_project/jst-rag-demo/jst-rag-demo/测试文件/金盘benchmark测试.xlsx",
                       help='批量测试输入Excel文件路径')
    parser.add_argument('--output', type=str, help='批量测试输出JSONL文件路径，如果不提供则使用时间戳命名')
    parser.add_argument('--config', type=str, help='包含检索和生成参数的JSON字符串配置')
    parser.add_argument('--timestamp', action='store_true', help='输出文件名是否使用时间戳命名')
    return parser.parse_args()

def run_single_query(query: str) -> None:
    """运行单条查询并直接输出结果"""
    print(f"处理查询: {query}")
    print("-" * 50)

    try:
        # 调用检索和回答生成
        search_response = route_and_search(
            query=query,
            # topk=30,  # 已移除
            alpha=0.5,  # BM25权重α
            pre_topk=30,  # BM25+Embedding预选
            faiss_per_index=50,  # 每子库向量检索topK
            bm25_per_index=50,  # 每子库BM25检索topK
            rerank_topk=30,  # 重排返回topK
            rerank_instruct="Given a web search query, retrieve relevant passages that answer the query.",
            neighbor_radius=1,  # 返回上下相邻chunk半径
            # return_table_full=True,  # 命中表格返回整表
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
                answer_messages,processed_contexts = build_answer_messages(
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

def run_benchmark_test(input_xlsx: str, output_jsonl: str, config: Dict[str, Any] = None) -> None:
    """运行基准测试"""
    if config is None:
        config = {}

    # 读取测试问题
    print("读取测试数据...")
    df = pd.read_excel(input_xlsx)
    questions = df.iloc[:, 0].tolist()  # 第一列是问题

    processed_count = 0  # 已处理数量计数
    total_time = 0.0  # 总耗时

    # 打开输出文件（覆盖模式，如果需要追加可以改为'a'）
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for i, question in enumerate(questions, 1):
            question = question.strip()  # 去掉首尾空白字符，包括换行符
            print(f"处理问题 {i}/{len(questions)}: {question[:50]}...")
            # 输出进度信息到stderr，供前端实时显示
            print(f"PROGRESS: 测试 {i}/{len(questions)}", file=sys.stderr, flush=True)

            start_time = datetime.now()  # 记录开始时间

            try:
                # 调用检索和回答生成
                search_response = route_and_search(
                    query=question,
                    # topk=30,  # 已移除
                    alpha=config.get('alpha', 0.5),  # BM25权重α
                    pre_topk=config.get('pre_topk', 30),  # BM25+Embedding预选
                    faiss_per_index=config.get('faiss_per_index', 50),  # 每子库向量检索topK
                    bm25_per_index=config.get('bm25_per_index', 50),  # 每子库BM25检索topK
                    rerank_topk=config.get('rerank_topk', 10),  # 重排返回topK
                    rerank_instruct="Given a web search query, retrieve relevant passages that answer the query.",
                    neighbor_radius=config.get('neighbor_radius', 1),  # 返回上下相邻chunk半径
                    # return_table_full=True,  # 命中表格返回整表
                    prefer_year=True,
                    doc_type_override=None,  # 自动识别
                    run_faiss=config.get('run_faiss', True),
                    run_bm25=config.get('run_bm25', True),
                )

                # 提取意图信息
                intent_dict = search_response.get("intent", {})
                # 提取检索结果列表（使用明确的变量名避免冲突）
                retrieved_chunks = search_response.get("results", [])

                # 生成回答
                model_answer = ""
                gen_answer = config.get('gen_answer', True)
                
                if gen_answer and retrieved_chunks:
                    contexts_for_answer = retrieved_chunks  # route_and_search 已经截断到 rerank_topk
                    try:
                        answer_messages,processed_contexts = build_answer_messages(
                            query=question,
                            contexts=contexts_for_answer,
                            system_prompt="你是严谨的中文金融助理。基于给定检索片段回答，若无法确定，请明确说明。引用用 [编号] 标注。",
                            per_chunk_limit=1200,  # 每条上下文最大字符数
                            include_full_table=True,
                        )
                        
                        chat_model = config.get('answer_model') or QWEN_CHAT_MODEL
                        if not chat_model.strip():
                            chat_model = QWEN_CHAT_MODEL
                            
                        model_answer = call_chat_completion(
                            messages=answer_messages,
                            model=chat_model,
                            temperature=config.get('answer_temperature', 0.1),  # 回答temperature
                            max_tokens=config.get('answer_max_tokens', 512),   # 回答max_tokens
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

                # 计算耗时
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()

                # 添加耗时到结果中
                single_result["处理耗时(秒)"] = processing_time
                total_time += processing_time

                # 立即写入结果到文件
                json_line = json.dumps(single_result, ensure_ascii=False)
                f.write(json_line + '\n')
                f.flush()  # 确保立即写入磁盘
                processed_count += 1

                print(f"✓ 已保存结果 {processed_count}/{len(questions)} (耗时: {processing_time:.2f}秒)")

            except Exception as e:
                # 计算耗时（即使失败也要记录）
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                total_time += processing_time

                print(f"处理问题失败: {e}")
                error_result = {
                    "问题": question,
                    "大模型回答": f"处理失败: {str(e)}",
                    "识别出的年份": "",
                    "识别出的财报/公告": "",
                    "处理耗时(秒)": processing_time
                }
                # 立即写入错误结果到文件
                json_line = json.dumps(error_result, ensure_ascii=False)
                f.write(json_line + '\n')
                f.flush()  # 确保立即写入磁盘
                processed_count += 1

                print(f"✗ 已保存错误结果 {processed_count}/{len(questions)} (耗时: {processing_time:.2f}秒)")

    # 计算平均耗时
    avg_time_per_question = total_time / processed_count if processed_count > 0 else 0.0

    print(f"\n测试完成，结果保存到: {output_jsonl}")
    print(f"共处理 {processed_count} 条结果")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"平均每个问题耗时: {avg_time_per_question:.2f} 秒")

def evaluate_answer_similarity(question: str, standard_answer: str, model_answer: str) -> float:
    """使用大模型评估两个答案的相似性，返回0-1之间的分数"""
    if not standard_answer or not model_answer:
        return 0.0

    try:
        evaluation_prompt = f"""请根据问题、标准答案、模型答案，评估模型答案的事实准确性。
结合标准答案，判断模型答案是否能正确回答问题。不要求完全与标准答案相似，因为标准答案也是大模型生成的，不一定准确。
关于数字类问题：无需考虑单位是否相同，此外标准答案中可能是经过四舍五入的数字，只需要模型答案中结果四舍五入符合标准答案，即算作完全正确。
关于文字类问题：无需考虑表达方式，文字长短，判断标准回答中的要点模型答案是否答出即可，如果答出要点则视作完全正确。
如果模型回答有部分要点符合标准回答，则按比例给分。
问题：{question}

标准答案：{standard_answer}

模型答案：{model_answer}

请只返回一个0-1之间的数字，表示得分。0表示完全错误，1表示回答正确。
请直接返回数字，不要其他任何文字。"""

        messages = [
            {"role": "system", "content": "你是一位老师，正在批阅学生的试卷，请严格按照要求输出。"},
            {"role": "user", "content": evaluation_prompt}
        ]

        evaluation_result = call_chat_completion(
            messages=messages,
            model=QWEN_CHAT_MODEL,
            temperature=0.1,  # 低温度保证评估稳定
            max_tokens=10,    # 只返回数字
        )

        # 提取数字
        match = re.search(r'0?\.\d+|\d+\.?\d*', evaluation_result.strip())
        if match:
            score = float(match.group())
            # 确保分数在0-1范围内
            return max(0.0, min(1.0, score))
        else:
            print(f"无法解析评估结果: {evaluation_result}")
            return 0.0

    except Exception as e:
        print(f"评估相似性失败: {str(e)}")
        return 0.0


def run_evaluation(input_jsonl: str, standard_xlsx: str, output_xlsx: str) -> None:
    """运行评估并生成结果表格"""
    print("开始评估...")

    # 读取测试结果
    results = []
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    # 读取标准答案
    df_standard = pd.read_excel(standard_xlsx)

    # 确保有标准回答列
    if '标准回答' not in df_standard.columns:
        raise ValueError("标准答案文件中没有'标准回答'列")

    # 为结果添加评估分数
    evaluation_results = []
    total_evaluation_time = 0.0

    for i, result in enumerate(results):
        question = result.get('问题', '')
        model_answer = result.get('大模型回答', '')

        # 找到对应的标准答案
        standard_row = df_standard[df_standard['问题'] == question]
        if len(standard_row) == 0:
            print(f"警告：问题 '{question[:50]}...' 没有找到标准答案")
            standard_answer = ""
        else:
            standard_answer = str(standard_row['标准回答'].iloc[0])
            

        # 计算相似性分数并记录耗时
        eval_start_time = datetime.now()
        similarity_score = evaluate_answer_similarity(question,standard_answer, model_answer)
        eval_end_time = datetime.now()
        evaluation_time = (eval_end_time - eval_start_time).total_seconds()
        total_evaluation_time += evaluation_time

        evaluation_result = {
            **result,
            '标准回答': standard_answer,
            '相似性分数': similarity_score,
            '评估耗时(秒)': evaluation_time
        }
        evaluation_results.append(evaluation_result)
        print(f"评估进度: {i+1}/{len(results)}, 相似性分数: {similarity_score:.3f}, 评估耗时: {evaluation_time:.2f}秒")
        # 输出进度信息到stderr，供前端实时显示
        print(f"PROGRESS: 评估 {i+1}/{len(results)}", file=sys.stderr, flush=True)

    # 保存为Excel文件
    df_results = pd.DataFrame(evaluation_results)
    df_results.to_excel(output_xlsx, index=False, engine='openpyxl')

    # 计算平均分数和平均评估耗时
    avg_score = df_results['相似性分数'].mean() if len(df_results) > 0 else 0.0
    avg_evaluation_time = total_evaluation_time / len(results) if len(results) > 0 else 0.0

    print(f"平均相似性分数: {avg_score:.3f}")
    print(f"平均评估耗时: {avg_evaluation_time:.2f} 秒")
    print(f"评估结果已保存到: {output_xlsx}")


def run_benchmark_test_with_timestamp(input_xlsx: str, config: Dict[str, Any] = None) -> str:
    """运行基准测试，输出文件名带时间戳"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_jsonl = f"/Users/wangyaqi/Documents/cursor_project/jst-rag-demo/jst-rag-demo/测试文件/benchmark_results_{timestamp}.jsonl"

    run_benchmark_test(input_xlsx, output_jsonl, config=config)
    return output_jsonl


if __name__ == "__main__":
    args = parse_arguments()

    # 解析配置
    config = {}
    if args.config:
        try:
            config = json.loads(args.config)
            print("已加载配置参数:")
            print(json.dumps(config, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"解析配置参数失败: {e}")
            sys.exit(1)

    if args.query:
        # 执行单条查询
        run_single_query(args.query)
    else:
        # 执行批量测试
        if args.output is None or args.timestamp:
            # 使用时间戳命名
            output_file = run_benchmark_test_with_timestamp(args.input, config=config)
        else:
            # 使用指定的输出文件
            run_benchmark_test(args.input, args.output, config=config)
            output_file = args.output

        print(f"\n输出文件: {output_file}")

        # 如果是时间戳模式，自动运行评估
        if args.output is None or args.timestamp:
            evaluation_xlsx = output_file.replace('.jsonl', '_evaluation.xlsx')
            try:
                run_evaluation(output_file, args.input, evaluation_xlsx)
            except Exception as e:
                print(f"评估失败: {str(e)}")

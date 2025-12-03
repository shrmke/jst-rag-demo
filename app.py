#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import shlex
import subprocess
import textwrap
import select
import pandas as pd
import base64
from io import BytesIO
from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st

from search import route_and_search as route_and_search_e2e
from search import route_and_search_multiquery_hyde as route_and_search_mq_hyde

SEARCH_PY = "/home/wangyaqi/jst/search.py"
DEFAULT_FAISS_FIN = "/home/wangyaqi/jst/金盘财报_indexes/faiss_exp"
DEFAULT_BM25_FIN = "/home/wangyaqi/jst/金盘财报_indexes/bm25_exp"
DEFAULT_FAISS_ANN = "/home/wangyaqi/jst/金盘上市公告_indexes/faiss_exp"
DEFAULT_BM25_ANN = "/home/wangyaqi/jst/金盘上市公告_indexes/bm25_exp"


def build_cmd(
    query: str,
    use_kb_finance: bool,
    faiss_dir: str,
    bm25_dir: str,
    pre_topk: int,
    rerank_topk: int,
    faiss_per_index: int,
    bm25_per_index: int,
    alpha: float,
    neighbor_radius: int,
    return_table_full: bool,
    gen_answer: bool,
    answer_topk: int,
    answer_temperature: float,
    answer_max_tokens: int,
    answer_model: Optional[str],
    quiet_results: bool = True,
) -> List[str]:
    cmd: List[str] = ["python3", SEARCH_PY]
    if use_kb_finance:
        cmd += ["--kb", "财报"]
    else:
        if faiss_dir:
            cmd += ["--faiss-dir", faiss_dir]
        if bm25_dir:
            cmd += ["--bm25-dir", bm25_dir]
    cmd += [
        "--query", query,
        "--pre-topk", str(pre_topk),
        "--rerank-topk", str(rerank_topk),
        "--faiss-per-index", str(faiss_per_index),
        "--bm25-per-index", str(bm25_per_index),
        "--alpha", str(alpha),
        "--neighbor-radius", str(neighbor_radius),
        "--mode", "hybrid",
        "--variant", "exp",
    ]
    if return_table_full:
        cmd.append("--return-table-full")
    if gen_answer:
        cmd.append("--gen-answer")
        cmd += [
            "--answer-topk", str(answer_topk),
            "--answer-temperature", str(answer_temperature),
            "--answer-max-tokens", str(answer_max_tokens),
        ]
        if answer_model:
            cmd += ["--answer-model", answer_model]
    if quiet_results:
        cmd.append("--quiet-results")
    return cmd


def run_search(cmd: List[str]) -> Dict[str, Any]:
    # 继承服务器环境变量（包含 DASHSCOPE_API_KEY/DASHSCOPE_BASE_URL）
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=os.environ.copy(),
    )
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    if proc.returncode != 0:
        raise RuntimeError(f"search.py 运行失败（代码 {proc.returncode}）\nSTDERR:\n{stderr}\nSTDOUT:\n{stdout}")
    # 预期为单行 JSON；若多行，取最后一个可解析 JSON
    last_json = None
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            last_json = json.loads(line)
            break
        except Exception:
            continue
    if last_json is None:
        # 尝试整体解析
        try:
            last_json = json.loads(stdout)
        except Exception as e:
            raise RuntimeError(f"无法解析 search.py 输出为 JSON：{e}\n原始输出:\n{stdout}") from e
    return last_json


def render_results(results: List[Dict[str, Any]], show_header: bool = True) -> None:
    if show_header:
        st.subheader("检索结果")
    if not results:
        st.info("未检索到结果")
        return
    for i, r in enumerate(results, start=1):
        with st.expander(f"[{i}] {r.get('doc_id','')} | page={r.get('page')} | type={r.get('type')}"):
            st.write(f"id: {r.get('id')}")
            st.write(f"order: {r.get('order')}")
            st.write(f"index_dir: {r.get('index_dir','')}")
            st.write(f"source_file: {r.get('source_file','')}")
            table = r.get("table")
            if isinstance(table, dict) and table:
                st.caption("命中表格，展示整表/行内容：")
            content = r.get("content") or ""
            if isinstance(content, str):
                st.code(content, language=None)
            else:
                st.json(content)


def render_trace(trace: Dict[str, Any], intent: Optional[Dict[str, Any]] = None) -> None:
    st.subheader("意图识别与路由过程")
    if not isinstance(trace, dict):
        st.info("无 trace 数据")
        return

    # 显示步骤进度
    stages = trace.get("stages") or []
    total_stages = len(stages)
    if total_stages > 0:
        st.caption(f"总共 {total_stages} 个步骤")

        # 显示每个步骤的进度
        for i, s in enumerate(stages, start=1):
            name = s.get("name")
            dur = s.get("duration_ms")
            # 根据步骤名给出中文描述
            step_descriptions = {
                "rules.detect_complexity": "检测查询复杂度",
                "rules.classify_doc_type": "识别文档类型",
                "intent.finalize": "完成意图整合"
            }
            desc = step_descriptions.get(name, name)
            with st.expander(f"[{i}/{total_stages}] {desc} ({dur} ms)"):
                s_show = dict(s)
                tokens = s_show.pop("tokens", None)
                if s_show:
                    st.json(s_show)
                if isinstance(tokens, dict):
                    st.caption("Tokens")
                    st.json(tokens)

        # 显示总耗时
        totals = trace.get("totals") or {}
        if totals:
            st.caption("总体统计")
            st.json(totals)


def render_answer(answer: str, sources: List[Dict[str, Any]]) -> None:
    st.subheader("回答")
    st.write(answer or "")
    st.subheader("引用来源")
    if not sources:
        st.info("无引用来源")
        return
    for s in sources:
        ref = s.get("ref")
        page = s.get("page")
        src_path = s.get("source_file") or ""
        modal = s.get("modal") or "unknown"
        base = os.path.basename(src_path) if isinstance(src_path, str) else ""
        # 仅展示原文件名、页码与 chunk 内容
        st.markdown(f"[{ref}] 文件：`{base}` | 页：`{page}` | 类型：`{modal}`")
        # 找到与该 source 相同 id 的内容无法直接从 sources 中取，需要前端按需求自行补全；
        # 此处假定 search.py 在 sources 中不含 content，因此容错从可能的字段取
        content = s.get("content") or s.get("snippet") or ""
        if isinstance(content, str) and content.strip():
            st.code(content, language=None)
        else:
            st.caption("此来源未附带内容片段。")


def sidebar_help():
    with st.sidebar.expander("使用说明", expanded=False):
        st.markdown(
            textwrap.dedent(
                """
                1. 服务器上运行（已部署向量/检索）：
                   - `export DASHSCOPE_API_KEY=你的Key`
                   - `export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"`
                   - `streamlit run app.py --server.address 0.0.0.0 --server.port 8501`

                2. 本地访问方式：
                   - 推荐 SSH 隧道：
                     `ssh -N -L 8501:127.0.0.1:8501 <用户名>@<服务器IP>`
                     然后浏览器打开 `http://localhost:8501`
                   - 或开放服务器 8501 端口，浏览器访问 `http://<服务器IP>:8501`

                3. 知识库：
                   - 选择“财报”将自动映射至:
                     - FAISS: /home/wangyaqi/jst/金盘财报_indexes/faiss_exp
                     - BM25:  /home/wangyaqi/jst/金盘财报_indexes/bm25_exp
                   - 选择“自定义路径”可手动指定。
                """
            )
        )


def create_download_link(df: pd.DataFrame, filename: str) -> str:
    """创建下载链接"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    output.seek(0)

    b64 = base64.b64encode(output.read()).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">下载 {filename}</a>'
    return href


def run_benchmark_evaluation(params: Dict[str, Any], progress_callback=None) -> tuple[pd.DataFrame, Dict[str, float]]:
    """运行基准测试和评估"""
    try:
        # 构建命令
        cmd = [
            "python3", "/home/wangyaqi/jst/测试文件/test_benchmark.py",
            "--input", params["input_file"],
            "--timestamp"  # 使用时间戳命名
        ]

        # 运行测试，使用Popen以便实时读取输出
        proc = subprocess.Popen(
            cmd,
            cwd="/home/wangyaqi",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=os.environ.copy(),
        )

        # 实时读取stderr中的进度信息
        progress_info = {"current": 0, "total": 0, "stage": "测试"}
        while True:
            # 非阻塞读取stderr
            if proc.stderr and select.select([proc.stderr], [], [], 0.1)[0]:
                line = proc.stderr.readline().strip()
                if line.startswith("PROGRESS: "):
                    progress_text = line.replace("PROGRESS: ", "")
                    if "测试" in progress_text:
                        parts = progress_text.replace("测试 ", "").split("/")
                        if len(parts) == 2:
                            progress_info["current"] = int(parts[0])
                            progress_info["total"] = int(parts[1])
                            progress_info["stage"] = "测试"
                    elif "评估" in progress_text:
                        parts = progress_text.replace("评估 ", "").split("/")
                        if len(parts) == 2:
                            progress_info["current"] = int(parts[0])
                            progress_info["total"] = int(parts[1])
                            progress_info["stage"] = "评估"

                    # 调用进度回调函数
                    if progress_callback:
                        progress_callback(progress_info)

            # 检查进程是否结束
            if proc.poll() is not None:
                break

        # 获取完整的输出
        stdout, stderr = proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(f"测试失败：{stderr}")

        # 从输出中提取信息
        output_lines = stdout.strip().split('\n')
        jsonl_file = None
        total_time = 0.0
        avg_time_per_question = 0.0
        avg_evaluation_time = 0.0

        for line in output_lines:
            if line.startswith("输出文件: "):
                jsonl_file = line.replace("输出文件: ", "").strip()
            elif line.startswith("总耗时: "):
                total_time = float(line.replace("总耗时: ", "").replace(" 秒", ""))
            elif line.startswith("平均每个问题耗时: "):
                avg_time_per_question = float(line.replace("平均每个问题耗时: ", "").replace(" 秒", ""))
            elif line.startswith("平均评估耗时: "):
                avg_evaluation_time = float(line.replace("平均评估耗时: ", "").replace(" 秒", ""))

        if not jsonl_file:
            raise RuntimeError("无法找到输出文件路径")

        # 读取评估结果
        evaluation_xlsx = jsonl_file.replace('.jsonl', '_evaluation.xlsx')
        if os.path.exists(evaluation_xlsx):
            df_results = pd.read_excel(evaluation_xlsx)

            # 返回结果和统计信息
            stats = {
                "总耗时": total_time,
                "平均问题耗时": avg_time_per_question,
                "平均评估耗时": avg_evaluation_time
            }

            return df_results, stats
        else:
            raise RuntimeError("评估结果文件未生成")

    except Exception as e:
        raise RuntimeError(f"运行测试失败: {str(e)}")


def render_benchmark_page():
    """渲染自动测评页面"""
    st.title("自动测评系统")

    # 参数设置区域
    st.header("参数设置")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("检索参数")
        pre_topk = st.number_input("pre_topk（BM25+Embedding预选）", min_value=5, max_value=200, value=30, step=1, key="bench_pre_topk")
        rerank_topk = st.number_input("rerank_topk（重排返回）", min_value=1, max_value=50, value=10, step=1, key="bench_rerank_topk")
        faiss_per_index = st.number_input("每子库向量检索 topK", min_value=1, max_value=100, value=50, step=1, key="bench_faiss_per_index")
        bm25_per_index = st.number_input("每子库 BM25 检索 topK", min_value=1, max_value=200, value=50, step=1, key="bench_bm25_per_index")
        alpha = st.slider("BM25 权重 α（混合打分）", min_value=0.0, max_value=1.0, value=0.5, step=0.05, key="bench_alpha")
        neighbor_radius = st.number_input("返回上下相邻 chunk 半径", min_value=0, max_value=5, value=1, step=1, key="bench_neighbor_radius")
        return_table_full = st.checkbox("命中表格返回整表", value=True, key="bench_return_table_full")

    with col2:
        st.subheader("回答参数")
        gen_answer = st.checkbox("生成回答", value=True, key="bench_gen_answer")
        answer_topk = st.number_input("用于回答的TopK条数", min_value=1, max_value=50, value=10, step=1, key="bench_answer_topk")
        answer_temperature = st.slider("回答 temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05, key="bench_answer_temperature")
        answer_max_tokens = st.number_input("回答 max_tokens", min_value=128, max_value=4096, value=512, step=64, key="bench_answer_max_tokens")
        answer_model = st.text_input("回答模型（可选）", value="", key="bench_answer_model")

        st.subheader("测试文件")
        input_file = st.text_input("基准测试文件路径", value="/home/wangyaqi/jst/测试文件/金盘benchmark测试.xlsx", key="bench_input_file")

    # 运行测试按钮
    if st.button("开始自动测试", type="primary", use_container_width=True):
        params = {
            "pre_topk": pre_topk,
            "rerank_topk": rerank_topk,
            "faiss_per_index": faiss_per_index,
            "bm25_per_index": bm25_per_index,
            "alpha": alpha,
            "neighbor_radius": neighbor_radius,
            "return_table_full": return_table_full,
            "gen_answer": gen_answer,
            "answer_topk": answer_topk,
            "answer_temperature": answer_temperature,
            "answer_max_tokens": answer_max_tokens,
            "answer_model": answer_model,
            "input_file": input_file
        }

        # 创建进度条和状态显示
        progress_bar = st.progress(0)
        progress_text = st.empty()
        status_text = st.empty()

        with st.status("正在运行基准测试...", expanded=True) as status:
            try:
                # 定义进度回调函数
                def update_progress(progress_info):
                    if progress_info["total"] > 0:
                        progress = progress_info["current"] / progress_info["total"]
                        progress_bar.progress(progress)
                        progress_text.text(f"{progress_info['stage']}进度: {progress_info['current']}/{progress_info['total']}")
                        status_text.text(f"正在{progress_info['stage']}第 {progress_info['current']} 个问题...")

                # 运行测试和评估
                df_results, stats = run_benchmark_evaluation(params, progress_callback=update_progress)

                # 保存到session_state
                st.session_state.benchmark_results = df_results
                st.session_state.benchmark_stats = stats
                st.session_state.benchmark_params = params
                st.session_state.benchmark_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # 更新进度条为完成状态
                progress_bar.progress(1.0)
                progress_text.text("测试完成")
                status_text.text("所有问题处理完毕")

                status.update(label="测试完成", state="complete")

                st.success("测试完成！")

            except Exception as e:
                status.update(label="测试失败", state="error")
                st.error(str(e))
                return

    # 显示结果
    if "benchmark_results" in st.session_state:
        st.header("测试结果")

        df_results = st.session_state.benchmark_results
        stats = st.session_state.get("benchmark_stats", {})
        params = st.session_state.benchmark_params
        timestamp = st.session_state.benchmark_timestamp

        # 显示参数摘要
        st.subheader("测试参数")
        param_summary = {
            "时间": timestamp,
            "pre_topk": params["pre_topk"],
            "rerank_topk": params["rerank_topk"],
            "alpha": params["alpha"],
            "neighbor_radius": params["neighbor_radius"],
            "用于回答的TopK": params["answer_topk"],
            "回答temperature": params["answer_temperature"],
            "回答max_tokens": params["answer_max_tokens"]
        }
        st.json(param_summary)

        # 显示性能统计
        st.subheader("性能统计")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if '相似性分数' in df_results.columns:
                avg_score = df_results['相似性分数'].mean()
                st.metric("平均相似性分数", f"{avg_score:.3f}")
        with col2:
            if "平均问题耗时" in stats:
                st.metric("平均问题耗时", f"{stats['平均问题耗时']:.2f}", help="每个问题从检索到回答生成的平均耗时")
        with col3:
            if "平均评估耗时" in stats:
                st.metric("平均评估耗时", f"{stats['平均评估耗时']:.2f}", help="评估每个答案相似性的平均耗时")
        with col4:
            if "总耗时" in stats:
                st.metric("总耗时", f"{stats['总耗时']:.1f}", help="整个测试过程的总耗时")

        # 显示结果表格
        st.subheader("详细结果")
        st.dataframe(df_results, use_container_width=True)

        # 下载按钮
        st.subheader("下载结果")
        filename = f"benchmark_results_{timestamp}.xlsx"
        st.markdown(create_download_link(df_results, filename), unsafe_allow_html=True)


def main():
    # 页面选择
    st.set_page_config(page_title="金盘知识库系统", layout="wide")

    page = st.sidebar.selectbox("选择页面", ["检索与问答", "自动测评"])

    if page == "检索与问答":
        render_search_page()
    else:
        render_benchmark_page()


def render_search_page():
    """渲染检索与问答页面（原main函数内容）"""
    st.title("金盘知识库 检索与问答")
    sidebar_help()

    # --- 侧边栏参数 ---
    st.sidebar.header("知识库")
    # 当启用 e2e（意图识别与年份路由）时，自动判断类型与年份，忽略知识库选择
    use_e2e_enabled = st.session_state.get("use_e2e", True)
    faiss_dir = ""
    bm25_dir = ""
    use_kb_finance = False
    use_kb_announce = False
    if use_e2e_enabled:
        st.sidebar.info("已启用 e2e：系统将自动识别财报/公告与年份进行路由，忽略知识库选择。")
    else:
        kb_mode = st.sidebar.radio("选择知识库", options=["财报", "公告", "自定义路径"], index=0, horizontal=True)
        use_kb_finance = kb_mode == "财报"
        use_kb_announce = kb_mode == "公告"
        if use_kb_announce:
            # 公告库：固定为上市公告索引根
            st.sidebar.caption("公告库将使用：金盘上市公告_indexed/")
            st.sidebar.write("FAISS 根目录：", DEFAULT_FAISS_ANN)
            st.sidebar.write("BM25 根目录：", DEFAULT_BM25_ANN)
            faiss_dir = DEFAULT_FAISS_ANN
            bm25_dir = DEFAULT_BM25_ANN
        elif not use_kb_finance:
            faiss_dir = st.sidebar.text_input("FAISS 目录", value=DEFAULT_FAISS_FIN)
            bm25_dir = st.sidebar.text_input("BM25 目录", value=DEFAULT_BM25_FIN)

    st.sidebar.header("检索/重排参数")
    pre_topk = st.sidebar.number_input("pre_topk（BM25+Embedding预选）", min_value=5, max_value=200, value=30, step=1)
    rerank_topk = st.sidebar.number_input("rerank_topk（重排返回）", min_value=1, max_value=50, value=10, step=1)
    faiss_per_index = st.sidebar.number_input("每子库向量检索 topK", min_value=1, max_value=100, value=50, step=1)
    bm25_per_index = st.sidebar.number_input("每子库 BM25 检索 topK", min_value=1, max_value=200, value=50, step=1)
    alpha = st.sidebar.slider("BM25 权重 α（混合打分）", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    neighbor_radius = st.sidebar.number_input("返回上下相邻 chunk 半径", min_value=0, max_value=5, value=1, step=1)
    return_table_full = st.sidebar.checkbox("命中表格返回整表", value=True)

    st.sidebar.header("文档类型指定")
    doc_type_choice = st.sidebar.selectbox(
        "若不指定则自动识别",
        options=["自动识别", "仅财报", "仅公告", "财报+公告"],
        index=0,
    )
    doc_type_override = None
    if doc_type_choice == "仅财报":
        doc_type_override = "report"
    elif doc_type_choice == "仅公告":
        doc_type_override = "notice"
    elif doc_type_choice == "财报+公告":
        doc_type_override = "both"

    st.sidebar.header("Multi-Query / HyDE")
    use_mq_hyde = st.sidebar.checkbox("启用 Multi-Query + HyDE", value=False)
    per_query_topk = st.sidebar.number_input("每路 topK（per_query_topk）", min_value=1, max_value=50, value=5, step=1)
    use_multiquery = st.sidebar.checkbox("使用 Multi-Query", value=False)
    use_hyde = st.sidebar.checkbox("使用 HyDE", value=False)
    mq_parallel = st.sidebar.checkbox("多路检索并行执行", value=False)

    st.sidebar.header("回答参数")
    gen_answer = st.sidebar.checkbox("生成回答", value=True)
    answer_topk = st.sidebar.number_input("用于回答的TopK条数", min_value=1, max_value=50, value=10, step=1)
    answer_temperature = st.sidebar.slider("回答 temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    answer_max_tokens = st.sidebar.number_input("回答 max_tokens", min_value=128, max_value=4096, value=512, step=64)
    answer_max_chars = st.sidebar.number_input("每条上下文最大字符数", min_value=200, max_value=5000, value=1200, step=100)
    answer_model = st.sidebar.text_input("回答模型（可选，留空用默认）", value="")

    # --- 主区 ---
    st.checkbox("启用意图识别与年份路由（e2e）", value=True, key="use_e2e")
    query = st.text_area("请输入问题", value="", height=120, placeholder="例如：2021年第一季度公司总资产是多少？")
    col1, col2 = st.columns([1, 5])
    with col1:
        run_btn = st.button("运行", type="primary", use_container_width=True)
    with col2:
        st.caption("提示：生成回答将调用大模型。若只想看召回结果，取消“生成回答”。")

    if run_btn:
        if not query.strip():
            st.warning("请输入问题")
            return
        if st.session_state.get("use_e2e"):
            # 直接调用 e2e 路由检索
            with st.status("意图识别与路由检索中…", expanded=False) as status:
                try:
                    if use_mq_hyde:
                        # 使用 multi-query + HyDE 的并行检索
                        data = route_and_search_mq_hyde(
                            query=query.strip(),
                            per_query_topk=int(per_query_topk),
                            use_multiquery=bool(use_multiquery),
                            use_hyde=bool(use_hyde),
                            parallel=bool(mq_parallel),
                            alpha=float(alpha),
                            pre_topk=int(pre_topk),
                            faiss_per_index=int(faiss_per_index),
                            bm25_per_index=int(bm25_per_index),
                            rerank_topk=int(rerank_topk),
                            neighbor_radius=int(neighbor_radius),
                            return_table_full=bool(return_table_full),
                            prefer_year=True,
                            doc_type_override=doc_type_override,
                        )
                    else:
                        data = route_and_search_e2e(
                            query=query.strip(),
                            topk=int(answer_topk),  # 取回答用 topk 作为主检索 topk
                            alpha=float(alpha),
                            pre_topk=int(pre_topk),
                            faiss_per_index=int(faiss_per_index),
                            bm25_per_index=int(bm25_per_index),
                            rerank_topk=int(rerank_topk),
                            neighbor_radius=int(neighbor_radius),
                            return_table_full=bool(return_table_full),
                            prefer_year=True,
                            doc_type_override=doc_type_override,
                        )
                    status.update(label="完成", state="complete", expanded=False)
                except Exception as e:
                    status.update(label="失败", state="error", expanded=True)
                    st.error(str(e))
                    return
            # 展示结果
            if use_mq_hyde:
                # 展示意图
                intent = (data or {}).get("intent") or {}
                with st.expander("意图识别结果", expanded=True):
                    st.json(intent)
                # 展示 Multi-Query 与 HyDE
                st.subheader("扩展查询（Multi-Query）")
                expansions = (data or {}).get("expansions") or {}
                if expansions:
                    for k in sorted(expansions.keys()):
                        st.markdown(f"- 方法 {k}")
                        st.code(str(expansions.get(k) or ""), language=None)
                else:
                    st.caption("无扩展查询或已关闭。")
                st.subheader("HyDE 模拟文档")
                hyde_text = (data or {}).get("hyde_passage") or ""
                if isinstance(hyde_text, str) and hyde_text.strip():
                    st.code(hyde_text.strip(), language=None)
                else:
                    st.caption("未生成 HyDE 文档或已关闭。")
                # 分路检索结果
                st.subheader("分路检索结果（原始/扩展/HyDE）")
                per_results = (data or {}).get("per_query_results") or {}
                if per_results:
                    for key in per_results.keys():
                        items = per_results.get(key) or []
                        with st.expander(f"{key}（{len(items)}）", expanded=False):
                            render_results(items, show_header=False)
                # 合并结果
                merged = (data or {}).get("merged_results") or []
                with st.expander(f"检索结果（合并去重后，{len(merged)} 条）", expanded=False):
                    render_results(merged, show_header=False)
                # 生成回答（用合并结果）
                if gen_answer:
                    used = (merged or [])[: max(1, int(answer_topk))]
                    from search import build_answer_messages, call_chat_completion, QWEN_CHAT_MODEL
                    messages = build_answer_messages(
                        query=query.strip(),
                        contexts=used,
                        system_prompt=st.session_state.get("answer_system", "你是严谨的中文金融助理。基于给定检索片段回答，若无法确定，请明确说明。引用用 [编号] 标注。"),
                        per_chunk_limit=int(answer_max_chars),
                        include_full_table=True,
                    )
                    chat_model = (answer_model.strip() or None) or QWEN_CHAT_MODEL
                    ans = call_chat_completion(
                        messages=messages,
                        model=chat_model,
                        temperature=float(answer_temperature),
                        max_tokens=int(answer_max_tokens),
                    )
                    sources = []
                    for i, it in enumerate(used, start=1):
                        sources.append({
                            "ref": i,
                            "id": it.get("id"),
                            "doc_id": it.get("doc_id"),
                            "page": it.get("page"),
                            "type": it.get("type"),
                            "order": it.get("order"),
                            "content": it.get("content"),
                            "index_dir": it.get("index_dir"),
                            "source_file": it.get("source_file"),
                        })
                    render_answer(ans, sources)
            else:
                # 原有 e2e 展示路径
                intent = (data or {}).get("intent") or {}
                trace = (data or {}).get("trace") or {}
                with st.expander("意图识别结果", expanded=True):
                    st.json(intent)
                render_trace(trace, intent=intent)
                results = (data or {}).get("results") or []
                if gen_answer:
                    used = results[: max(1, int(answer_topk))]
                    from search import build_answer_messages, call_chat_completion, QWEN_CHAT_MODEL
                    messages,processed_contexts = build_answer_messages(
                        query=query.strip(),
                        contexts=used,
                        system_prompt=st.session_state.get("answer_system", "你是严谨的中文金融助理。基于给定检索片段回答，若无法确定，请明确说明。引用用 [编号] 标注。"),
                        per_chunk_limit=int(answer_max_chars),
                        include_full_table=True,
                    )
                    chat_model = (answer_model.strip() or None) or QWEN_CHAT_MODEL
                    ans = call_chat_completion(
                        messages=messages,
                        model=chat_model,
                        temperature=float(answer_temperature),
                        max_tokens=int(answer_max_tokens),
                    )
                    sources = []
                    for i, it in enumerate(used, start=1):
                        sources.append({
                            "ref": i,
                            "id": it.get("id"),
                            "doc_id": it.get("doc_id"),
                            "page": it.get("page"),
                            "type": it.get("type"),
                            "order": it.get("order"),
                            "content": it.get("content"),
                            "index_dir": it.get("index_dir"),
                            "source_file": it.get("source_file"),
                        })
                    render_answer(ans, processed_contexts)
        else:
            # 保留原命令行分支
            cmd = build_cmd(
                query=query.strip(),
                use_kb_finance=use_kb_finance,
                faiss_dir=faiss_dir.strip(),
                bm25_dir=bm25_dir.strip(),
                pre_topk=int(pre_topk),
                rerank_topk=int(rerank_topk),
                faiss_per_index=int(faiss_per_index),
                bm25_per_index=int(bm25_per_index),
                alpha=float(alpha),
                neighbor_radius=int(neighbor_radius),
                return_table_full=bool(return_table_full),
                gen_answer=bool(gen_answer),
                answer_topk=int(answer_topk),
                answer_temperature=float(answer_temperature),
                answer_max_tokens=int(answer_max_tokens),
                answer_model=answer_model.strip() or None,
                quiet_results=True,
            )
            with st.status("检索中，请稍候…", expanded=False) as status:
                try:
                    st.write("命令：", " ".join([shlex.quote(x) for x in cmd]))
                    data = run_search(cmd)
                    status.update(label="检索完成", state="complete", expanded=False)
                except Exception as e:
                    status.update(label="检索失败", state="error", expanded=True)
                    st.error(str(e))
                    return
            # 渲染
            if gen_answer and isinstance(data, dict) and "answer" in data:
                render_answer(answer=data.get("answer", ""), sources=data.get("sources") or [])
            else:
                st.caption("检索已完成。未开启“生成回答”，因此不展示结果列表。")


if __name__ == "__main__":
    main()



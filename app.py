#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import shlex
import subprocess
import textwrap
from typing import Any, Dict, List, Optional

import streamlit as st

from search import route_and_search as route_and_search_e2e

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


def render_results(results: List[Dict[str, Any]]) -> None:
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
    # 进度条（外层：子问题；内层：年份）
    subq_total = 1
    year_total = 0
    if isinstance(intent, dict):
        subq_total = max(1, len(intent.get("sub_questions") or []))
        years = intent.get("years") or []
        year_total = len(years)
    with st.container():
        st.caption("进度（外层：子问题；内层：年份）")
        outer = st.progress(1.0 if subq_total == 0 else 1.0)  # 同步渲染，已完成
        inner = st.progress(1.0 if year_total == 0 else 1.0)   # 同步渲染，已完成
        st.caption(f"子问题：{subq_total} / {subq_total}，年份：{year_total} / {year_total}")
    # 阶段时间线
    stages = trace.get("stages") or []
    for i, s in enumerate(stages, start=1):
        name = s.get("name")
        dur = s.get("duration_ms")
        with st.expander(f"[{i}] {name} ({dur} ms)"):
            s_show = dict(s)
            tokens = s_show.pop("tokens", None)
            if s_show:
                st.json(s_show)
            if isinstance(tokens, dict):
                st.caption("Tokens")
                st.json(tokens)
    totals = trace.get("totals") or {}
    if totals:
        st.caption("Totals")
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
        base = os.path.basename(src_path) if isinstance(src_path, str) else ""
        # 仅展示原文件名、页码与 chunk 内容
        st.markdown(f"[{ref}] 文件：`{base}` | 页：`{page}`")
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


def main():
    st.set_page_config(page_title="检索与问答 | Streamlit", layout="wide")
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
                        output_cap=200,
                        prefer_year=True,
                    )
                    status.update(label="完成", state="complete", expanded=False)
                except Exception as e:
                    status.update(label="失败", state="error", expanded=True)
                    st.error(str(e))
                    return
            # 展示意图与过程
            intent = (data or {}).get("intent") or {}
            trace = (data or {}).get("trace") or {}
            with st.expander("意图识别结果", expanded=True):
                st.json(intent)
            render_trace(trace, intent=intent)
            # 结果与可选回答
            results = (data or {}).get("results") or []
            if gen_answer:
                used = results[: max(1, int(answer_topk))]
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
                # 将 used 组装为 sources
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



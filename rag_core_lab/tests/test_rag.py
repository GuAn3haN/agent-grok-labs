import sys
import os

# 将项目根目录（rag_core_lab）加入 Python 模块搜索路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.generator import RAGGenerator
from src.retriever import BaseRetriever
from src.chunker import SimpleChunker

def test_no_rag_baseline():

    class EmptyRetriever(BaseRetriever):
        def index(self, chunks): pass
        def retrieve(self, query, k=3): return []

    gen = RAGGenerator(EmptyRetriever())
    answer = gen.answer("XJ-2024-007 号钻探报告的核心结论是什么？")
    assert "XJ-2024-007" not in answer, f"LLM 不可能知道这个编号，但幻觉输出了：{answer}"

def test_rag_basic():
    docs = [
        "XJ-2024-007 号钻探报告显示，矿区深部存在厚层花岗岩基底。",
        "2023年公司营收为12亿元，同比增长8%。"
    ]
    retriever = BaseRetriever()
    retriever.index(docs)
    gen = RAGGenerator(retriever)
    result = gen.answer("XJ-2024-007 钻探报告发现了什么？", return_context=True)
    assert "花岗岩基底" in result["answer"], f"答案未包含事实：{result['answer']}"
    assert len(result["chunks"]) > 0

def test_chunk_size_impact():
    text = "XJ-2024-007 号钻探报告指出，矿区深部存在厚层花岗岩基底，这是本次勘探的主要发现。"
    small_chunker = SimpleChunker(chunk_size=20, chunk_overlap=0)
    small_chunks = small_chunker.split_text(text)
    assert len(small_chunks) > 1
    for ch in small_chunks:
        assert not ("钻探" in ch and "花岗岩基底" in ch), "小块不应同时包含两者"

    large_chunker = SimpleChunker(chunk_size=200, chunk_overlap=0)
    large_chunks = large_chunker.split_text(text)
    combined = any("钻探" in ch and "花岗岩基底" in ch for ch in large_chunks)
    assert combined, "大块应能包含完整事实"

def test_k_value_impact():
    chunks = [
        "XJ-2024-008 报告：新钻探技术效率提升 20%。",
        "同时，该技术使每米钻进成本降低 12%。"
    ]
    retriever = BaseRetriever()
    retriever.index(chunks)
    gen = RAGGenerator(retriever)

    ans_k1 = gen.answer("新钻探技术提升了多少效率，成本有何变化？", k=1, return_context=True)
    ans_k2 = gen.answer("新钻探技术提升了多少效率，成本有何变化？", k=2, return_context=True)
    assert "12%" not in ans_k1["answer"], f"K=1 时意外包含了成本信息：{ans_k1}"
    assert "12%" in ans_k2["answer"], f"K=2 时应包含成本信息：{ans_k2}"



def test_source_traceability():
    chunks = [
        "F-2024-01: 单晶高温合金可在 1100°C 工作。",
        "F-2024-02: 该合金抗氧化性能提升 30%。"
    ]

    retriever = BaseRetriever()
    retriever.index(chunks)
    gen = RAGGenerator(retriever)
    result = gen.answer("F-2024-01 研究中的合金耐温多少？", return_context=True)

    assert "1100" in result["answer"]                    # 答案必须包含温度数值
    assert any("1100°C" in ch for ch in result["chunks"]) # 返回的块中至少有一个包含该数值

def test_rag_vs_llm_hallucination():
    doc = "XJ-2024-009 号勘探区发现高品位锂矿，氧化锂平均品位 1.8%。"
    retriever = BaseRetriever()
    retriever.index([doc])
    gen_rag = RAGGenerator(retriever)

    class EmptyRetriever(BaseRetriever):
        def index(self, chunks): pass
        def retrieve(self, query, k=3): return []
    gen_no_rag = RAGGenerator(EmptyRetriever())

    ans_rag = gen_rag.answer("XJ-2024-009 的主要发现是什么？")
    ans_no_rag = gen_no_rag.answer("XJ-2024-009 的主要发现是什么？")
    assert "1.8%" in ans_rag, f"RAG 答案应包含 1.8%，实际为：{ans_rag}"
    assert "1.8%" not in ans_no_rag, f"纯 LLM 不应知道 1.8%，实际为：{ans_no_rag}"

def test_multi_doc_synthesis():
    chunks = [
        "报告 A: 该地区年降水量增加 12%。",
        "报告 B: 同期平均气温上升 1.2°C。"
    ]
    retriever = BaseRetriever()
    retriever.index(chunks)
    gen = RAGGenerator(retriever)
    ans = gen.answer("该地区气候变化的关键数据有哪些？", k=2)
    assert "12%" in ans and "1.2°C" in ans, f"答案应同时包含两个数据，实际为：{ans}"
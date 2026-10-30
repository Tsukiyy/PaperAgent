import json
from datetime import UTC, datetime

import pytest

from paper_agent.config import Settings
from paper_agent.exceptions import SynthesisError
from paper_agent.models import (
    ComparisonRow,
    DocumentChunk,
    DownloadedPdf,
    EvidenceClaim,
    EvidenceReference,
    Paper,
    PaperEvidence,
    ParsedDocument,
    SectionRange,
)
from paper_agent.services.synthesis import (
    ClaimSynthesisService,
    LLMClaimSynthesizer,
    TemplateClaimSynthesizer,
    build_structured_comparison,
)
from paper_agent.tools.research import _write_eval


def make_analysis(index: int) -> PaperEvidence:
    paper_id = f"paper-{index}"
    now = datetime(2025, 1, index, tzinfo=UTC)
    paper = Paper(
        paper_id=paper_id,
        title=f"Paper {index}",
        authors=("Researcher",),
        abstract="Abstract.",
        published_at=now,
        updated_at=now,
        paper_url=f"https://example.com/{paper_id}",
        pdf_url=f"https://example.com/{paper_id}.pdf",
    )
    reference = EvidenceReference(
        chunk_id=f"p{index}-c1",
        page_number=index,
        section_title="Methods",
        quote=f"Method evidence {index}.",
    )
    document = ParsedDocument(
        paper_id=paper_id,
        pdf_path=f"{paper_id}.pdf",
        page_count=index,
        sections=(SectionRange(title="Methods", start_page=1, end_page=index),),
        chunks=(
            DocumentChunk(
                chunk_id=reference.chunk_id,
                page_number=index,
                section_title="Methods",
                text=reference.quote,
            ),
        ),
    )
    return PaperEvidence(
        paper=paper,
        pdf=DownloadedPdf(
            paper_id=paper_id,
            source_url=paper.pdf_url,
            local_path=f"{paper_id}.pdf",
            sha256="a" * 64,
            byte_size=100,
        ),
        document=document,
        claims=(
            EvidenceClaim(
                paper_id=paper_id,
                claim=f"Paper {index} uses method {index}.",
                claim_type="method",
                evidence=(reference,),
            ),
            EvidenceClaim(
                paper_id=paper_id,
                claim=f"Paper {index} reports result {index}.",
                claim_type="result",
                evidence=(reference,),
            ),
        ),
        provider="template",
    )


def make_comparison(analysis: PaperEvidence) -> ComparisonRow:
    return ComparisonRow(
        paper_id=analysis.paper.paper_id,
        title=analysis.paper.title,
        paper_url=analysis.paper.paper_url,
        methods="aggregated method text",
        results="aggregated result text",
        evidence_locations="p.1 §Methods",
    )


def test_structured_comparison_uses_order_and_claims_from_analyses() -> None:
    first = make_analysis(1)
    second = make_analysis(2)
    records = build_structured_comparison(
        (first, second),
        (make_comparison(second), make_comparison(first)),
    )

    assert [record["paper_id"] for record in records] == ["paper-2", "paper-1"]
    assert records[0]["paper_ref"] == "P1"
    assert records[0]["compared_dimensions"] == ["method", "result"]
    assert records[0]["claims_by_dimension"]["method"][0]["claim"] == ("Paper 2 uses method 2.")
    assert records[0]["claims_by_dimension"]["method"][0]["evidence"] == [
        {"page": 2, "section": "Methods", "quote": "Method evidence 2."}
    ]
    assert records[0]["abstract"] == "Abstract."


def test_template_synthesis_exposes_horizontal_review_sections() -> None:
    analyses = (make_analysis(1), make_analysis(2))
    comparisons = tuple(make_comparison(analysis) for analysis in analyses)

    text = TemplateClaimSynthesizer().generate("RAG", analyses, comparisons)

    assert "## 分论文详细分析" in text
    assert "### 论文共识" in text
    assert "### 方法差异" in text
    assert "### 数据集差异" in text
    assert "### 评价指标差异" in text
    assert "### 实验设置差异" in text
    assert "## 实验结果与可比性" in text
    assert "[P1, p.1 §Methods] Paper 1 uses method 1." in text
    assert "[P2, p.2 §Methods] Paper 2 reports result 2." in text


def test_eval_claim_coverage_counts_claims_with_any_valid_evidence(tmp_path) -> None:
    analysis = make_analysis(1)
    claim = analysis.claims[0]
    analysis = analysis.model_copy(
        update={"claims": (claim.model_copy(update={"evidence": claim.evidence * 2}),)}
    )
    path = tmp_path / "eval_summary.md"

    _write_eval(path, "run-1", (analysis.paper,), (analysis,), (), (), 1, True)

    content = path.read_text(encoding="utf-8")
    assert "证据引用数：2" in content
    assert "至少有一条有效证据的 claim 数：1" in content
    assert "claim 证据覆盖率：100.00%" in content


def test_forced_llm_synthesis_reports_missing_configuration() -> None:
    service = ClaimSynthesisService(
        Settings(
            _env_file=None,
            LLM_API_KEY=None,
            LLM_BASE_URL=None,
            LLM_MODEL=None,
            PAPER_AGENT_SYNTHESIS_PROVIDER="llm",
        )
    )

    with pytest.raises(SynthesisError, match="LLM_API_KEY.*LLM_BASE_URL.*LLM_MODEL"):
        service.generate("RAG", (), ())


def test_llm_synthesis_receives_structured_claim_json(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeCompletions:
        def create(self, *, model: str, messages, max_tokens: int, extra_body):
            captured.update(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )
            message = type("Message", (), {"content": "### 总体综合\n有证据的综合。"})()
            choice = type(
                "Choice",
                (),
                {"message": message, "finish_reason": "stop"},
            )()
            return type("Response", (), {"choices": [choice]})()

    class FakeClient:
        chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(
        "paper_agent.services.synthesis.create_llm_client",
        lambda *, api_key, base_url, proxy=None: FakeClient(),
    )
    analysis = make_analysis(1)
    text = LLMClaimSynthesizer(
        api_key="test-key",
        base_url="https://api.moonshot.cn/v1",
        model="kimi-k2.6",
        max_tokens=4096,
        proxy="http://127.0.0.1:7890",
    ).generate("RAG", (analysis,), (make_comparison(analysis),))
    messages = captured["messages"]
    records = json.loads(messages[1]["content"].split("结构化论文对比 JSON：\n", 1)[1])

    assert text == "### 总体综合\n有证据的综合。"
    assert captured["model"] == "kimi-k2.6"
    assert captured["max_tokens"] == 4096
    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}
    assert records[0]["claims_by_dimension"]["method"][0]["claim"] == ("Paper 1 uses method 1.")
    assert records[0]["claims_by_dimension"]["method"][0]["evidence"] == [
        {"page": 1, "section": "Methods", "quote": "Method evidence 1."}
    ]

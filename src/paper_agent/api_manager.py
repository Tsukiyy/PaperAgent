"""研究任务的后台执行与结果适配。"""

import json
import sqlite3
import threading
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from paper_agent.clients.arxiv import ArxivClient
from paper_agent.clients.openalex import OpenAlexClient
from paper_agent.clients.semantic_scholar import SemanticScholarClient
from paper_agent.config import get_settings
from paper_agent.evidence.extractor import EvidenceService
from paper_agent.services.multi_source_search import MultiSourceSearchService
from paper_agent.services.research_workflow import ResearchWorkflow, create_run_paths
from paper_agent.services.synthesis import ClaimSynthesisService
from paper_agent.services.trace import RunTrace
from paper_agent.storage.database import SQLiteStore
from paper_agent.storage.vector_index import FaissVectorIndex
from paper_agent.tools.pdf import PdfDownloader, PyMuPdfParser
from paper_agent.tools.research import ResearchToolContext


@dataclass
class RunRecord:
    run_id: str
    query: str
    limit: int
    read_limit: int
    status: str = "queued"
    stage: str = "queued"
    message: str = "任务已排队"
    error: str | None = None
    result: dict[str, Any] | None = None
    output_dir: Path | None = None


@dataclass
class ResearchRunManager:
    """在进程内管理后台研究任务；单机 MVP 不引入额外队列服务。"""

    records: dict[str, RunRecord] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def create(self, query: str, limit: int, read_limit: int) -> RunRecord:
        settings = get_settings()
        run_id, run_dir, _, _ = create_run_paths(settings.output_dir, settings.data_dir)
        record = RunRecord(run_id, query, limit, read_limit, output_dir=run_dir)
        with self.lock:
            self.records[run_id] = record
        self._persist(record)
        return record

    def get(self, run_id: str) -> RunRecord | None:
        with self.lock:
            record = self.records.get(run_id)
        if record is not None:
            return record
        history = self.get_history(run_id)
        if history is None:
            return None
        record = RunRecord(
            run_id=history["run_id"],
            query=history["query"],
            limit=history["limit"],
            read_limit=history["read_limit"],
            status=history["status"],
            stage=history["stage"],
            message=history["message"],
            error=history["error"],
            result=history["result"],
            output_dir=Path(history["output_dir"]),
        )
        with self.lock:
            self.records[run_id] = record
        return record

    @staticmethod
    def _store() -> SQLiteStore:
        settings = get_settings()
        return SQLiteStore(settings.data_dir / "paper_agent.db")

    def _persist(self, record: RunRecord) -> None:
        with self._store() as store:
            store.save_research_history(
                run_id=record.run_id,
                query=record.query,
                limit=record.limit,
                read_limit=record.read_limit,
                status=record.status,
                stage=record.stage,
                message=record.message,
                error=record.error,
                result=record.result,
                output_dir=str(record.output_dir or ""),
            )

    def list_history(
        self, *, query: str = "", status: str = "", limit: int = 50, offset: int = 0
    ) -> dict[str, Any]:
        with self._store() as store:
            items, total = store.list_research_history(
                query=query, status=status, limit=limit, offset=offset
            )
        return {"items": items, "total": total, "limit": limit, "offset": offset}

    def get_history(self, run_id: str) -> dict[str, Any] | None:
        with self._store() as store:
            return store.get_research_history(run_id)

    def initialize(self) -> None:
        self.import_legacy_outputs()
        with self._store() as store:
            store.mark_interrupted_research_runs()

    def import_legacy_outputs(self) -> None:
        """将已有 CLI 运行目录索引到历史表，保留产物原文件。"""

        settings = get_settings()
        output_root = settings.output_dir
        database_path = settings.data_dir / "paper_agent.db"
        if not output_root.exists():
            return
        for run_dir in output_root.iterdir():
            if not run_dir.is_dir() or not (run_dir / "review.md").is_file():
                continue
            run_id = run_dir.name
            if self.get_history(run_id) is not None:
                continue
            result = self._legacy_result(run_dir, database_path)
            if result is None:
                continue
            query = result["query"]
            with self._store() as store:
                store.save_research_history(
                    run_id=run_id,
                    query=query,
                    limit=result["summary"]["paper_count"] or 1,
                    read_limit=result["summary"]["read_count"] or 1,
                    status="completed",
                    stage="completed",
                    message="已导入既有运行产物",
                    output_dir=str(run_dir.resolve()),
                    result=result,
                )

    def _legacy_result(self, run_dir: Path, database_path: Path) -> dict[str, Any] | None:
        review = self._read_text(run_dir / "review.md")
        if not review:
            return None
        run_id = run_dir.name
        query = review.splitlines()[0].lstrip("# ").strip() or "历史研究任务"
        papers: list[dict[str, Any]] = []
        if database_path.is_file():
            try:
                with sqlite3.connect(database_path) as connection:
                    run = connection.execute(
                        "SELECT query FROM runs WHERE run_id = ?", (run_id,)
                    ).fetchone()
                    if run and run[0]:
                        query = run[0]
                    rows = connection.execute(
                        """SELECT p.paper_id, p.title, p.authors_json, p.published_at,
                                  p.paper_url, p.citation_count, p.is_open_access,
                                  COALESCE((SELECT source FROM paper_sources s
                                            WHERE s.paper_id=p.paper_id LIMIT 1), 'unknown')
                           FROM run_papers rp JOIN papers p ON p.paper_id=rp.paper_id
                           WHERE rp.run_id=? ORDER BY rp.rank""",
                        (run_id,),
                    ).fetchall()
                papers = [
                    {
                        "paper_id": row[0],
                        "title": row[1],
                        "authors": json.loads(row[2]),
                        "published_at": row[3],
                        "paper_url": row[4],
                        "source": row[7],
                        "citation_count": row[5],
                        "is_open_access": bool(row[6]),
                    }
                    for row in rows
                ]
            except sqlite3.Error:
                papers = []
        claims_by_paper: dict[str, dict[str, Any]] = {}
        evidence_path = run_dir / "evidence.jsonl"
        if evidence_path.is_file():
            for line in evidence_path.read_text(encoding="utf-8").splitlines():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                paper_id = item.get("paper_id")
                if not paper_id:
                    continue
                paper = claims_by_paper.setdefault(
                    paper_id,
                    {
                        "paper_id": paper_id,
                        "title": item.get("title", paper_id),
                        "paper_url": item.get("paper_url", ""),
                        "provider": item.get("provider", "unknown"),
                        "groups": {},
                    },
                )
                claim_type = item.get("claim_type", "other")
                paper["groups"].setdefault(claim_type, []).append(
                    {"claim": item.get("claim", ""), "references": item.get("evidence", [])}
                )
        type_names = {
            "contribution": "研究贡献", "method": "方法", "dataset": "数据集",
            "metric": "评价指标", "result": "实验结果", "limitation": "局限性",
            "other": "其他",
        }
        evidence_groups = [
            {
                **{key: value for key, value in paper.items() if key != "groups"},
                "groups": [
                    {"type": key, "label": type_names.get(key, key), "claims": claims}
                    for key, claims in paper["groups"].items()
                ],
            }
            for paper in claims_by_paper.values()
        ]
        eval_markdown = self._read_text(run_dir / "eval_summary.md")
        claim_count = sum(
            len(group["claims"])
            for paper in evidence_groups
            for group in paper["groups"]
        )
        review_passed = "证据不足" not in eval_markdown and "失败" not in eval_markdown
        return {
            "run_id": run_id,
            "query": query,
            "summary": {
                "review_passed": review_passed,
                "paper_count": len(papers),
                "read_count": len(evidence_groups),
                "evidence_count": claim_count,
                "failure_count": 0,
            },
            "papers": papers,
            "evidence_groups": evidence_groups,
            "comparisons": [],
            "failures": [],
            "review_markdown": review,
            "bibtex": self._read_text(run_dir / "papers.bib"),
            "eval_markdown": eval_markdown,
        }

    def run(self, run_id: str) -> None:
        record = self.get(run_id)
        if record is None:
            return
        self._update(record, status="running", stage="initializing", message="正在初始化研究服务")
        settings = get_settings()
        run_dir = settings.output_dir / run_id
        pdf_dir = settings.data_dir / "papers"
        faiss_dir = settings.data_dir / "faiss" / run_id
        try:
            with ExitStack() as stack:
                arxiv = stack.enter_context(ArxivClient(timeout=settings.request_timeout))
                semantic_scholar = stack.enter_context(
                    SemanticScholarClient(
                        api_key=settings.semantic_scholar_api_key,
                        timeout=settings.request_timeout,
                        user_agent=settings.semantic_scholar_user_agent,
                        proxy=settings.http_proxy,
                    )
                )
                openalex = stack.enter_context(
                    OpenAlexClient(
                        api_key=settings.openalex_api_key,
                        timeout=settings.request_timeout,
                        proxy=settings.http_proxy,
                    )
                )
                downloader = stack.enter_context(
                    PdfDownloader(
                        timeout=settings.request_timeout,
                        max_bytes=settings.max_pdf_bytes,
                        proxy=settings.http_proxy,
                    )
                )
                store = stack.enter_context(SQLiteStore(settings.data_dir / "paper_agent.db"))
                context = ResearchToolContext(
                    search_service=MultiSourceSearchService(
                        {"arXiv": arxiv, "Semantic Scholar": semantic_scholar, "OpenAlex": openalex}
                    ),
                    downloader=downloader,
                    parser=PyMuPdfParser(),
                    evidence_service=EvidenceService(settings),
                    store=store,
                    vector_index=FaissVectorIndex(faiss_dir, dimension=settings.vector_dimension),
                    trace=RunTrace(),
                    pdf_dir=pdf_dir,
                    run_dir=run_dir,
                    synthesis_service=ClaimSynthesisService(settings),
                )
                self._update(record, stage="research", message="正在检索、解析论文并生成综述")
                result = ResearchWorkflow(context).run(
                    record.query, limit=record.limit, read_limit=record.read_limit
                )
            self._update(
                record,
                status="completed",
                stage="completed",
                message="研究任务已完成",
                result=self._format_result(result),
            )
        except Exception as exc:  # noqa: BLE001 - API 需要将任务错误反馈给前端
            self._update(
                record,
                status="failed",
                stage="failed",
                message="研究任务失败",
                error=str(exc),
            )

    def _update(self, record: RunRecord, **values: Any) -> None:
        with self.lock:
            for key, value in values.items():
                setattr(record, key, value)
            snapshot = RunRecord(**record.__dict__)
        self._persist(snapshot)

    @staticmethod
    def _read_text(path: Path) -> str:
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def _format_result(self, result: Any) -> dict[str, Any]:
        output_dir = Path(result.output_dir)
        settings = get_settings()
        evidence_groups: list[dict[str, Any]] = []
        type_names = {
            "contribution": "研究贡献",
            "method": "方法",
            "dataset": "数据集",
            "metric": "评价指标",
            "result": "实验结果",
            "limitation": "局限性",
            "other": "其他",
        }
        for analysis in result.analyses:
            claims_by_type: dict[str, list[dict[str, Any]]] = {}
            for claim in analysis.claims:
                claims_by_type.setdefault(claim.claim_type, []).append(
                    {
                        "claim": claim.claim,
                        "references": [ref.model_dump(mode="json") for ref in claim.evidence],
                    }
                )
            evidence_groups.append(
                {
                    "paper_id": analysis.paper.paper_id,
                    "title": analysis.paper.title,
                    "paper_url": str(analysis.paper.paper_url),
                    "provider": analysis.provider,
                    "groups": [
                        {"type": key, "label": type_names.get(key, key), "claims": claims}
                        for key, claims in claims_by_type.items()
                    ],
                }
            )
        papers = [paper.model_dump(mode="json") for paper in result.papers]
        process_methods = [
            {
                "stage": "论文检索与排序",
                "configured": "arXiv + Semantic Scholar + OpenAlex",
                "actual": "多源检索；确定性去重与启发式排序",
            },
            {
                "stage": "PDF 解析",
                "configured": "PyMuPDF",
                "actual": "PyMuPDF",
            },
            {
                "stage": "Evidence 提取",
                "configured": settings.evidence_provider,
                "actual": result.process.get("evidence_extraction", "unknown"),
            },
            {
                "stage": "Evidence 审核",
                "configured": (
                    f"{settings.review_provider}; review_llm_enabled="
                    f"{settings.review_llm_enabled}"
                ),
                "actual": result.process.get("evidence_review", "unknown"),
                "detail": result.process.get("evidence_review_detail", ""),
            },
            {
                "stage": "论文对比",
                "configured": "本地规则聚合",
                "actual": "规则聚合",
            },
            {
                "stage": "综述生成",
                "configured": settings.synthesis_provider,
                "actual": result.process.get("synthesis_generation", "unknown"),
                "detail": result.process.get("synthesis_generation_detail", ""),
            },
            {
                "stage": "综述引用审核",
                "configured": "本地规则",
                "actual": result.process.get("synthesis_review", "unknown"),
            },
        ]
        return {
            "run_id": result.run_id,
            "query": result.query,
            "summary": {
                "review_passed": result.review_passed,
                "paper_count": len(result.papers),
                "read_count": len(result.analyses),
                "evidence_count": sum(len(item.claims) for item in result.analyses),
                "failure_count": len(result.failures),
            },
            "papers": papers,
            "evidence_groups": evidence_groups,
            "comparisons": [item.model_dump(mode="json") for item in result.comparisons],
            "failures": [item.model_dump(mode="json") for item in result.failures],
            "process_methods": process_methods,
            "review_markdown": self._read_text(output_dir / "review.md"),
            "bibtex": self._read_text(output_dir / "papers.bib"),
            "eval_markdown": self._read_text(output_dir / "eval_summary.md"),
        }


manager = ResearchRunManager()


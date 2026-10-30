"""SQLite 主数据与证据存储。"""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from paper_agent.models import ComparisonRow, DownloadedPdf, Paper, PaperEvidence, ParsedDocument


class SQLiteStore:
    """维护论文、PDF、文本块、claims、证据与运行状态。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path)
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._create_schema()

    def __enter__(self) -> "SQLiteStore":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY, query TEXT NOT NULL, status TEXT NOT NULL,
                started_at TEXT NOT NULL, finished_at TEXT, error TEXT
            );
            CREATE TABLE IF NOT EXISTS papers (
                paper_id TEXT PRIMARY KEY, title TEXT NOT NULL, authors_json TEXT NOT NULL,
                abstract TEXT NOT NULL, published_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                paper_url TEXT NOT NULL, pdf_url TEXT, doi TEXT, external_ids_json TEXT NOT NULL,
                citation_count INTEGER NOT NULL, reference_count INTEGER NOT NULL,
                is_open_access INTEGER NOT NULL, rank_score REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS paper_sources (
                paper_id TEXT NOT NULL, source TEXT NOT NULL,
                PRIMARY KEY (paper_id, source),
                FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
            );
            CREATE TABLE IF NOT EXISTS run_papers (
                run_id TEXT NOT NULL, paper_id TEXT NOT NULL, rank INTEGER NOT NULL,
                PRIMARY KEY (run_id, paper_id),
                FOREIGN KEY (run_id) REFERENCES runs(run_id),
                FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
            );
            CREATE TABLE IF NOT EXISTS pdfs (
                sha256 TEXT PRIMARY KEY, paper_id TEXT NOT NULL, source_url TEXT NOT NULL,
                local_path TEXT NOT NULL, byte_size INTEGER NOT NULL,
                FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
            );
            CREATE TABLE IF NOT EXISTS chunks (
                paper_id TEXT NOT NULL, chunk_id TEXT NOT NULL, page_number INTEGER NOT NULL,
                section_title TEXT NOT NULL, text TEXT NOT NULL, pdf_path TEXT NOT NULL,
                PRIMARY KEY (paper_id, chunk_id),
                FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
            );
            CREATE TABLE IF NOT EXISTS claims (
                claim_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
                paper_id TEXT NOT NULL, claim_type TEXT NOT NULL, claim_text TEXT NOT NULL,
                provider TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id),
                FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
            );
            CREATE TABLE IF NOT EXISTS evidence (
                claim_id INTEGER NOT NULL, chunk_id TEXT NOT NULL, page_number INTEGER NOT NULL,
                section_title TEXT NOT NULL, quote TEXT NOT NULL,
                FOREIGN KEY (claim_id) REFERENCES claims(claim_id)
            );
            CREATE TABLE IF NOT EXISTS comparisons (
                run_id TEXT NOT NULL, paper_id TEXT NOT NULL, row_json TEXT NOT NULL,
                PRIMARY KEY (run_id, paper_id),
                FOREIGN KEY (run_id) REFERENCES runs(run_id),
                FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
            );
            CREATE TABLE IF NOT EXISTS research_run_history (
                run_id TEXT PRIMARY KEY, query TEXT NOT NULL, limit_count INTEGER NOT NULL,
                read_limit INTEGER NOT NULL, status TEXT NOT NULL, stage TEXT NOT NULL,
                message TEXT NOT NULL, error TEXT, result_json TEXT,
                output_dir TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            """
        )
        self._connection.commit()

    def save_research_history(
        self,
        *,
        run_id: str,
        query: str,
        limit: int,
        read_limit: int,
        status: str,
        stage: str,
        message: str,
        output_dir: str,
        error: str | None = None,
        result: dict | None = None,
    ) -> None:
        """保存 Web 任务索引、状态和完整结果快照。"""

        now = datetime.now(UTC).isoformat()
        self._connection.execute(
            """
            INSERT INTO research_run_history (
                run_id, query, limit_count, read_limit, status, stage, message,
                error, result_json, output_dir, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                status=excluded.status, stage=excluded.stage, message=excluded.message,
                error=excluded.error, result_json=excluded.result_json,
                output_dir=excluded.output_dir, updated_at=excluded.updated_at
            """,
            (
                run_id,
                query,
                limit,
                read_limit,
                status,
                stage,
                message,
                error,
                json.dumps(result, ensure_ascii=False) if result is not None else None,
                output_dir,
                now,
                now,
            ),
        )
        self._connection.commit()

    def list_research_history(
        self, *, query: str = "", status: str = "", limit: int = 50, offset: int = 0
    ) -> tuple[list[dict], int]:
        """分页读取历史记录索引，不加载大体积结果正文。"""

        clauses: list[str] = []
        params: list[object] = []
        if query.strip():
            clauses.append("query LIKE ?")
            params.append(f"%{query.strip()}%")
        if status:
            clauses.append("status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        total = self._connection.execute(
            f"SELECT COUNT(*) FROM research_run_history {where}", params
        ).fetchone()[0]
        rows = self._connection.execute(
            f"""
            SELECT run_id, query, limit_count, read_limit, status, stage, message,
                   error, output_dir, created_at, updated_at,
                   json_extract(result_json, '$.summary.paper_count'),
                   json_extract(result_json, '$.summary.read_count'),
                   json_extract(result_json, '$.summary.evidence_count')
            FROM research_run_history {where}
            ORDER BY created_at DESC LIMIT ? OFFSET ?
            """,
            [*params, limit, offset],
        ).fetchall()
        keys = (
            "run_id", "query", "limit", "read_limit", "status", "stage", "message",
            "error", "output_dir", "created_at", "updated_at", "paper_count",
            "read_count", "evidence_count",
        )
        return [dict(zip(keys, row, strict=True)) for row in rows], total

    def get_research_history(self, run_id: str) -> dict | None:
        """按任务 ID 读取历史索引和完整结果。"""

        row = self._connection.execute(
            """SELECT run_id, query, limit_count, read_limit, status, stage, message,
                      error, result_json, output_dir, created_at, updated_at
               FROM research_run_history WHERE run_id = ?""",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        keys = (
            "run_id", "query", "limit", "read_limit", "status", "stage", "message",
            "error", "result_json", "output_dir", "created_at", "updated_at",
        )
        item = dict(zip(keys, row, strict=True))
        item["result"] = json.loads(item.pop("result_json")) if item["result_json"] else None
        return item

    def mark_interrupted_research_runs(self) -> None:
        """服务启动时将上次进程遗留的活动任务标记为中断。"""

        now = datetime.now(UTC).isoformat()
        self._connection.execute(
            """UPDATE research_run_history
               SET status='failed', stage='interrupted', message='服务重启，任务中断',
                   error='服务重启时任务仍在运行', updated_at=?
               WHERE status IN ('queued', 'running')""",
            (now,),
        )
        self._connection.commit()

    def start_run(self, run_id: str, query: str) -> None:
        self._connection.execute(
            "INSERT INTO runs(run_id, query, status, started_at) VALUES (?, ?, ?, ?)",
            (run_id, query, "running", datetime.now(UTC).isoformat()),
        )
        self._connection.commit()

    def finish_run(self, run_id: str, *, status: str, error: str | None = None) -> None:
        self._connection.execute(
            "UPDATE runs SET status = ?, finished_at = ?, error = ? WHERE run_id = ?",
            (status, datetime.now(UTC).isoformat(), error, run_id),
        )
        self._connection.commit()

    def save_papers(self, run_id: str, papers: tuple[Paper, ...]) -> None:
        for rank, paper in enumerate(papers, start=1):
            self._connection.execute(
                """
                INSERT INTO papers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(paper_id) DO UPDATE SET
                    title=excluded.title, authors_json=excluded.authors_json,
                    abstract=excluded.abstract, paper_url=excluded.paper_url,
                    pdf_url=excluded.pdf_url, doi=excluded.doi,
                    external_ids_json=excluded.external_ids_json,
                    citation_count=excluded.citation_count,
                    reference_count=excluded.reference_count,
                    is_open_access=excluded.is_open_access, rank_score=excluded.rank_score
                """,
                (
                    paper.paper_id,
                    paper.title,
                    json.dumps(paper.authors, ensure_ascii=False),
                    paper.abstract,
                    paper.published_at.isoformat(),
                    paper.updated_at.isoformat(),
                    str(paper.paper_url),
                    str(paper.pdf_url) if paper.pdf_url else None,
                    paper.doi,
                    json.dumps(paper.external_ids, ensure_ascii=False, sort_keys=True),
                    paper.citation_count,
                    paper.reference_count,
                    int(paper.is_open_access),
                    paper.rank_score,
                ),
            )
            for source in paper.sources:
                self._connection.execute(
                    "INSERT OR IGNORE INTO paper_sources VALUES (?, ?)",
                    (paper.paper_id, source),
                )
            self._connection.execute(
                "INSERT OR REPLACE INTO run_papers VALUES (?, ?, ?)",
                (run_id, paper.paper_id, rank),
            )
        self._connection.commit()

    def save_pdf(self, pdf: DownloadedPdf) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO pdfs VALUES (?, ?, ?, ?, ?)",
            (pdf.sha256, pdf.paper_id, str(pdf.source_url), pdf.local_path, pdf.byte_size),
        )
        self._connection.commit()

    def save_document(self, document: ParsedDocument) -> None:
        self._connection.executemany(
            "INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    document.paper_id,
                    chunk.chunk_id,
                    chunk.page_number,
                    chunk.section_title,
                    chunk.text,
                    document.pdf_path,
                )
                for chunk in document.chunks
            ],
        )
        self._connection.commit()

    def save_analysis(self, run_id: str, analysis: PaperEvidence) -> None:
        for claim in analysis.claims:
            cursor = self._connection.execute(
                """INSERT INTO claims(run_id, paper_id, claim_type, claim_text, provider)
                VALUES (?, ?, ?, ?, ?)""",
                (run_id, claim.paper_id, claim.claim_type, claim.claim, analysis.provider),
            )
            claim_id = int(cursor.lastrowid or 0)
            self._connection.executemany(
                "INSERT INTO evidence VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        claim_id,
                        ref.chunk_id,
                        ref.page_number,
                        ref.section_title,
                        ref.quote,
                    )
                    for ref in claim.evidence
                ],
            )
        self._connection.commit()

    def save_comparisons(self, run_id: str, rows: tuple[ComparisonRow, ...]) -> None:
        self._connection.executemany(
            "INSERT OR REPLACE INTO comparisons VALUES (?, ?, ?)",
            [
                (run_id, row.paper_id, row.model_dump_json())
                for row in rows
            ],
        )
        self._connection.commit()

"""运行级 JSONL 审计追踪，不依赖 LangSmith。"""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter


class RunTrace:
    """按调用顺序记录工具或 Agent 节点的输入摘要、输出摘要和错误。"""

    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    @contextmanager
    def span(self, name: str, kind: str, inputs: dict[str, object]) -> Iterator[dict[str, object]]:
        started_at = datetime.now(UTC)
        start = perf_counter()
        record: dict[str, object] = {
            "name": name,
            "kind": kind,
            "started_at": started_at.isoformat(),
            "inputs": inputs,
        }
        try:
            yield record
            record["status"] = "ok"
        except Exception as exc:
            record["status"] = "error"
            record["error"] = str(exc)
            raise
        finally:
            record["finished_at"] = datetime.now(UTC).isoformat()
            record["duration_ms"] = round((perf_counter() - start) * 1000, 3)
            self.records.append(record)

    def write(self, path: Path) -> None:
        path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in self.records),
            encoding="utf-8",
        )

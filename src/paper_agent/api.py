"""PaperAgent 的 FastAPI 入口。"""

from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from paper_agent.api_manager import manager
from paper_agent.api_schemas import (
    ResearchCreateRequest,
    ResearchCreateResponse,
    ResearchStatusResponse,
)

app = FastAPI(title="PaperAgent API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
manager.initialize()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/research/runs", response_model=ResearchCreateResponse, status_code=202)
def create_research_run(
    request: ResearchCreateRequest, background_tasks: BackgroundTasks
) -> ResearchCreateResponse:
    record = manager.create(request.query, request.limit, request.read_limit)
    background_tasks.add_task(manager.run, record.run_id)
    return ResearchCreateResponse(run_id=record.run_id)


@app.get("/api/research/runs/{run_id}/status", response_model=ResearchStatusResponse)
def research_status(run_id: str) -> ResearchStatusResponse:
    record = manager.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="任务不存在或服务已重启。")
    return ResearchStatusResponse(
        run_id=record.run_id,
        status=record.status,
        stage=record.stage,
        message=record.message,
        error=record.error,
        result=record.result,
    )


@app.get("/api/research/runs/{run_id}/result")
def research_result(run_id: str) -> dict:
    record = manager.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="任务不存在或服务已重启。")
    if record.status != "completed" or record.result is None:
        raise HTTPException(status_code=409, detail="任务尚未完成。")
    return record.result


@app.get("/api/research/runs")
def research_history(
    q: str = Query(default="", max_length=200),
    status: str = Query(default="", pattern="^(|queued|running|completed|failed)$"),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict:
    """返回历史任务索引；详情通过 run_id 单独加载。"""

    return manager.list_history(query=q, status=status, limit=limit, offset=offset)


@app.get("/api/research/runs/{run_id}/files/{filename}")
def research_file(run_id: str, filename: str) -> FileResponse:
    """下载指定任务的原始产物，限制在允许的文件名集合内。"""

    allowed = {
        "review.md", "papers.bib", "evidence.jsonl", "comparison.csv",
        "trace.jsonl", "eval_summary.md",
    }
    if filename not in allowed:
        raise HTTPException(status_code=404, detail="不支持下载该文件。")
    history = manager.get_history(run_id)
    if history is None:
        raise HTTPException(status_code=404, detail="历史任务不存在。")
    path = Path(history["output_dir"]) / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="任务产物不存在。")
    return FileResponse(path, filename=filename)


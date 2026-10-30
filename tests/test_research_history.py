from paper_agent.storage.database import SQLiteStore


def test_research_history_persists_index_and_full_result(tmp_path):
    database = tmp_path / "paper_agent.db"
    result = {
        "query": "agentic RAG",
        "summary": {"paper_count": 2, "read_count": 1, "evidence_count": 4},
        "review_markdown": "# Review",
    }

    with SQLiteStore(database) as store:
        store.save_research_history(
            run_id="run-1",
            query="agentic RAG",
            limit=5,
            read_limit=2,
            status="completed",
            stage="completed",
            message="研究任务已完成",
            output_dir=str(tmp_path / "run-1"),
            result=result,
        )

    with SQLiteStore(database) as reopened:
        history = reopened.get_research_history("run-1")
        items, total = reopened.list_research_history(query="agentic", limit=10)

    assert history is not None
    assert history["result"] == result
    assert items[0]["run_id"] == "run-1"
    assert items[0]["paper_count"] == 2
    assert items[0]["evidence_count"] == 4
    assert total == 1


def test_research_history_can_filter_status(tmp_path):
    with SQLiteStore(tmp_path / "paper_agent.db") as store:
        for run_id, status in (("done", "completed"), ("failed", "failed")):
            store.save_research_history(
                run_id=run_id,
                query="topic",
                limit=3,
                read_limit=1,
                status=status,
                stage=status,
                message=status,
                output_dir=str(tmp_path / run_id),
            )
        items, total = store.list_research_history(status="failed")

    assert total == 1
    assert [item["run_id"] for item in items] == ["failed"]

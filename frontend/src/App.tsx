import { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type Status = "idle" | "queued" | "running" | "completed" | "failed";
type Claim = { claim: string; references: { page_number: number; section_title: string; quote: string }[] };
type EvidenceGroup = { type: string; label: string; claims: Claim[] };
type EvidencePaper = { paper_id: string; title: string; paper_url: string; provider: string; groups: EvidenceGroup[] };
type Paper = { paper_id: string; title: string; authors: string[]; published_at: string; paper_url: string; source: string; citation_count: number; is_open_access: boolean };
type Result = {
  run_id: string; query: string;
  summary: { review_passed: boolean; paper_count: number; read_count: number; evidence_count: number; failure_count: number };
  papers: Paper[]; evidence_groups: EvidencePaper[]; failures: { title: string; stage: string; reason: string }[];
  process_methods?: { stage: string; configured: string; actual: string; detail?: string }[];
  review_markdown: string; bibtex: string; eval_markdown: string;
};
type RunStatus = { run_id: string; status: Status; stage: string; message: string; error?: string | null; result?: Result | null };
type HistoryItem = { run_id: string; query: string; limit: number; read_limit: number; status: Status; stage: string; message: string; error?: string | null; created_at: string; updated_at: string; paper_count?: number; read_count?: number; evidence_count?: number };

const tabs = ["运行摘要", "Evidence", "论文", "综述", "BibTeX"] as const;
type Tab = (typeof tabs)[number];
const stageLabels: Record<string, string> = {
  queued: "任务排队", initializing: "初始化服务", research: "检索与分析", completed: "已完成", failed: "任务失败",
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, { ...init, headers: { "Content-Type": "application/json", ...init?.headers } });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const detail = body.detail;
    throw new Error(Array.isArray(detail) ? detail.map((item: { msg?: string }) => item.msg).join("；") : detail || `请求失败 (${response.status})`);
  }
  return response.json() as Promise<T>;
}

function App() {
  const [query, setQuery] = useState("");
  const [limit, setLimit] = useState(8);
  const [readLimit, setReadLimit] = useState(3);
  const [status, setStatus] = useState<RunStatus | null>(null);
  const [tab, setTab] = useState<Tab>("运行摘要");
  const [error, setError] = useState("");
  const [selectedPaper, setSelectedPaper] = useState("");
  const [filter, setFilter] = useState("全部类型");
  const [copied, setCopied] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [historyQuery, setHistoryQuery] = useState("");
  const [historyStatus, setHistoryStatus] = useState("");
  const [historyLoading, setHistoryLoading] = useState(false);
  const busy = status?.status === "queued" || status?.status === "running";
  const result = status?.result;

  async function loadHistory(search = historyQuery, statusFilter = historyStatus) {
    setHistoryLoading(true);
    try {
      const params = new URLSearchParams({ q: search, limit: "100" });
      if (statusFilter) params.set("status", statusFilter);
      const data = await request<{ items: HistoryItem[] }>(`/api/research/runs?${params.toString()}`);
      setHistory(data.items);
    } catch (err) {
      setError(err instanceof Error ? err.message : "无法加载历史任务");
    } finally {
      setHistoryLoading(false);
    }
  }

  async function openHistory(item: HistoryItem) {
    setError("");
    try {
      const detail = await request<RunStatus>(`/api/research/runs/${item.run_id}/status`);
      setStatus(detail);
      setTab("运行摘要");
      setSelectedPaper("");
      setShowHistory(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "无法加载历史任务详情");
    }
  }

  useEffect(() => {
    if (!status?.run_id || !busy) return;
    const timer = window.setInterval(async () => {
      try {
        const next = await request<RunStatus>(`/api/research/runs/${status.run_id}/status`);
        setStatus(next);
        if (next.status === "completed") setTab("运行摘要");
      } catch (err) {
        setError(err instanceof Error ? err.message : "读取任务状态失败");
      }
    }, 1200);
    return () => window.clearInterval(timer);
  }, [status?.run_id, busy]);

  useEffect(() => { void loadHistory(""); }, []);

  const evidencePapers = useMemo(() => {
    if (!result) return [];
    return selectedPaper ? result.evidence_groups.filter((paper) => paper.paper_id === selectedPaper) : result.evidence_groups;
  }, [result, selectedPaper]);

  async function startResearch(event: React.FormEvent) {
    event.preventDefault();
    setError("");
    setStatus(null);
    setTab("运行摘要");
    setSelectedPaper("");
    try {
      const created = await request<{ run_id: string; status: "queued" }>("/api/research/runs", {
        method: "POST", body: JSON.stringify({ query, limit, read_limit: readLimit }),
      });
      setStatus({ ...created, stage: "queued", message: "任务已提交", result: null });
    } catch (err) {
      setError(err instanceof Error ? err.message : "无法创建任务");
    }
  }

  async function copyBibtex() {
    if (!result) return;
    await navigator.clipboard.writeText(result.bibtex);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1800);
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="PaperAgent 首页"><span className="brand-mark">P</span><span>PaperAgent</span></a>
        <div className="topbar-meta"><span className="online-dot" />研究工作台<span className="topbar-divider" />多源论文研究</div>
        <div className="topbar-actions"><button className="history-button" onClick={() => { setShowHistory(true); void loadHistory(); }}>历史任务</button><button className="topbar-action" onClick={() => { setStatus(null); setError(""); setQuery(""); setShowHistory(false); }} disabled={busy}>＋ <span>新建研究</span></button></div>
      </header>

      {showHistory && <aside className="history-drawer"><div className="history-drawer-head"><div><p className="eyebrow">ARCHIVE <span>·</span> RESEARCH RUNS</p><h2>历史任务</h2></div><button className="drawer-close" onClick={() => setShowHistory(false)}>×</button></div><div className="history-search"><input value={historyQuery} onChange={(event) => setHistoryQuery(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter") void loadHistory(); }} placeholder="搜索研究主题" /><button onClick={() => void loadHistory()}>⌕</button></div><div className="history-filter"><span>状态</span><select value={historyStatus} onChange={(event) => { setHistoryStatus(event.target.value); void loadHistory(historyQuery, event.target.value); }}><option value="">全部</option><option value="completed">已完成</option><option value="failed">失败</option><option value="running">运行中</option><option value="queued">排队中</option></select><span className="history-count">{history.length} 条</span></div><div className="history-list">{historyLoading && <div className="history-empty">正在加载历史任务…</div>}{!historyLoading && history.length === 0 && <div className="history-empty">暂无历史任务</div>}{!historyLoading && history.map((item) => <button className="history-item" key={item.run_id} onClick={() => void openHistory(item)}><div className="history-item-top"><span className={`history-status ${item.status}`} /> <strong>{item.query}</strong><small>{formatDate(item.created_at)}</small></div><div className="history-item-meta"><span>{item.status === "completed" ? "已完成" : item.status === "failed" ? "失败" : item.message}</span>{item.paper_count != null && <span>{item.paper_count} 篇论文 · {item.evidence_count ?? 0} 条证据</span>}</div></button>)}</div></aside>}

      <main id="top" className="workspace">
        <section className="intro-row">
          <div><p className="eyebrow">RESEARCH WORKSPACE <span>·</span> 01</p><h1>从研究问题，到证据综述。</h1><p className="intro-copy">检索相关论文，追踪原文证据，并生成可复核的研究综述。</p></div>
          {status && <div className={`run-indicator ${status.status}`}><span className="status-dot" />{status.status === "completed" ? "运行已完成" : status.status === "failed" ? "运行失败" : stageLabels[status.stage] || status.message}</div>}
        </section>

        <section className="query-panel">
          <form onSubmit={startResearch}>
            <label className="field-label" htmlFor="query">研究主题</label>
            <div className="query-input-row"><input id="query" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="例如：agentic retrieval-augmented generation" maxLength={500} required disabled={busy} /><span className="input-hint">{query.length}/500</span></div>
            <div className="form-footer">
              <div className="number-fields">
                <label className="number-field"><span>检索论文</span><input type="number" min={1} max={20} value={limit} disabled={busy} onChange={(event) => { const value = Number(event.target.value); setLimit(value); if (readLimit > value) setReadLimit(value); }} /><small>篇</small></label>
                <span className="field-separator">/</span>
                <label className="number-field"><span>精读分析</span><input type="number" min={1} max={limit} value={readLimit} disabled={busy} onChange={(event) => setReadLimit(Number(event.target.value))} /><small>篇</small></label>
              </div>
              <button className="primary-button" type="submit" disabled={busy || !query.trim()}>{busy ? <><span className="spinner" />研究进行中</> : <>开始研究 <span>→</span></>}</button>
            </div>
          </form>
          {error && <div className="error-banner"><span>!</span>{error}</div>}
          {busy && <div className="progress-row"><div className="progress-copy"><span>{status?.message || "正在处理研究任务"}</span><span>{stageLabels[status?.stage || ""] || "工作流运行中"}</span></div><div className="progress-track"><span /></div></div>}
          {status?.status === "failed" && <div className="error-banner"><span>!</span>{status.error || "研究任务失败"}</div>}
        </section>

        {result ? <section className="results-shell">
          <div className="results-heading"><div><p className="eyebrow">RESULTS <span>·</span> {result.run_id}</p><h2>{result.query}</h2></div><span className={`review-badge ${result.summary.review_passed ? "passed" : "attention"}`}><span />{result.summary.review_passed ? "证据审核通过" : "需要复核"}</span></div>
          <nav className="tabs" aria-label="研究结果分类">{tabs.map((item) => <button key={item} onClick={() => setTab(item)} className={tab === item ? "active" : ""}>{item}{item === "Evidence" && <span className="tab-count">{result.summary.evidence_count}</span>}</button>)}</nav>

          {tab === "运行摘要" && <section className="result-body">
            <div className="metric-grid"><Metric label="检索论文" value={result.summary.paper_count} unit="篇" accent="green" /><Metric label="精读论文" value={result.summary.read_count} unit="篇" accent="blue" /><Metric label="Evidence" value={result.summary.evidence_count} unit="条" accent="coral" /><Metric label="处理失败" value={result.summary.failure_count} unit="篇" accent="slate" /></div>
            <div className="summary-columns"><div className="summary-block"><div className="block-heading"><h3>评测摘要</h3><span>RUN EVALUATION</span></div><div className="markdown-panel compact"><ReactMarkdown remarkPlugins={[remarkGfm]}>{result.eval_markdown || "暂无评测摘要。"}</ReactMarkdown></div></div><div className="summary-block"><div className="block-heading"><h3>论文处理概览</h3><span>{result.papers.length} PAPERS</span></div><div className="paper-overview">{result.papers.slice(0, 5).map((paper, index) => <div className="overview-paper" key={paper.paper_id}><span className="paper-index">{String(index + 1).padStart(2, "0")}</span><span className="overview-title">{paper.title}</span><span className={`paper-state ${result.evidence_groups.some((item) => item.paper_id === paper.paper_id) ? "ready" : "pending"}`}>{result.evidence_groups.some((item) => item.paper_id === paper.paper_id) ? "已分析" : "已检索"}</span></div>)}</div></div></div>
            {result.process_methods && <div className="process-methods"><div className="block-heading"><h3>本次运行方式</h3><span>CONFIGURED → ACTUAL</span></div><div className="process-method-list">{result.process_methods.map((item) => <div className="process-method-row" key={item.stage}><strong>{item.stage}</strong><span>{item.configured}</span><b>→</b><span className="actual-method"><em>{item.actual}</em>{item.detail && <small>{item.detail}</small>}</span></div>)}</div></div>}
            {result.failures.length > 0 && <div className="notice-block"><strong>部分论文未能完成分析</strong>{result.failures.map((failure, index) => <p key={`${failure.title}-${index}`}>{failure.title} · {failure.stage} · {failure.reason}</p>)}</div>}
          </section>}

          {tab === "Evidence" && <section className="result-body evidence-layout">
            <aside className="evidence-sidebar"><div className="side-title">精读论文 <span>{result.evidence_groups.length}</span></div><button className={`side-paper ${!selectedPaper ? "selected" : ""}`} onClick={() => setSelectedPaper("")}><span className="side-paper-dot all" /><span>全部论文</span><small>{result.summary.evidence_count}</small></button>{result.evidence_groups.map((paper) => <button key={paper.paper_id} className={`side-paper ${selectedPaper === paper.paper_id ? "selected" : ""}`} onClick={() => setSelectedPaper(paper.paper_id)}><span className="side-paper-dot" /><span>{paper.title}</span><small>{paper.groups.reduce((sum, group) => sum + group.claims.length, 0)}</small></button>)}</aside>
            <div className="evidence-content"><div className="evidence-toolbar"><div><h3>按论文整理的研究证据</h3><p>结论均保留对应原文、页码与章节位置</p></div><select value={filter} onChange={(event) => setFilter(event.target.value)}><option>全部类型</option>{[...new Set(result.evidence_groups.flatMap((paper) => paper.groups.map((group) => group.label)))].map((label) => <option key={label}>{label}</option>)}</select></div>
              {evidencePapers.map((paper) => <article className="evidence-paper" key={paper.paper_id}><div className="evidence-paper-heading"><div><span className="paper-kicker">{paper.paper_id}</span><h3><a href={paper.paper_url} target="_blank" rel="noreferrer">{paper.title} ↗</a></h3></div><span className="provider-label">{paper.provider}</span></div>{paper.groups.filter((group) => filter === "全部类型" || group.label === filter).map((group) => <div className="claim-group" key={group.type}><h4><span className={`claim-type-mark ${group.type}`} />{group.label}<small>{group.claims.length}</small></h4>{group.claims.map((claim, index) => <div className="claim-item" key={`${group.type}-${index}`}><p>{claim.claim}</p>{claim.references.map((ref, refIndex) => <blockquote key={`${ref.page_number}-${refIndex}`}><span className="quote-mark">“</span><span>{ref.quote}</span><cite>p. {ref.page_number} · {ref.section_title}</cite></blockquote>)}</div>)}</div>)}</article>)}
              {evidencePapers.length === 0 && <div className="empty-state">没有可展示的证据记录</div>}
            </div>
          </section>}

          {tab === "论文" && <section className="result-body"><div className="block-heading paper-list-title"><h3>检索结果</h3><span>{result.papers.length} PAPERS FOUND</span></div><div className="paper-list">{result.papers.map((paper, index) => <article className="paper-row" key={paper.paper_id}><span className="paper-index">{String(index + 1).padStart(2, "0")}</span><div className="paper-info"><h3><a href={paper.paper_url} target="_blank" rel="noreferrer">{paper.title} ↗</a></h3><p>{paper.authors.slice(0, 4).join(", ")}{paper.authors.length > 4 ? " 等" : ""} <span>·</span> {new Date(paper.published_at).getFullYear()} <span>·</span> {paper.source}</p><div className="paper-tags"><span>{paper.paper_id}</span>{paper.is_open_access && <span>开放获取</span>}{result.evidence_groups.some((item) => item.paper_id === paper.paper_id) && <span className="tag-ready">已完成精读</span>}</div></div><span className="citation-count">{paper.citation_count.toLocaleString()}<small>引用</small></span></article>)}</div></section>}

          {tab === "综述" && <section className="result-body document-view"><div className="document-toolbar"><div><span className="file-icon md">M</span><div><strong>review.md</strong><small>Markdown 综述 · 完整预览</small></div></div><button className="quiet-button" onClick={() => downloadText("review.md", result.review_markdown)}>↓ 下载 Markdown</button></div><div className="markdown-panel review-markdown"><ReactMarkdown remarkPlugins={[remarkGfm]}>{result.review_markdown || "暂无综述内容。"}</ReactMarkdown></div></section>}

          {tab === "BibTeX" && <section className="result-body document-view"><div className="document-toolbar"><div><span className="file-icon bib">B</span><div><strong>papers.bib</strong><small>论文引用 · BibTeX 格式</small></div></div><div className="document-actions"><button className="quiet-button" onClick={copyBibtex}>{copied ? "✓ 已复制" : "▢ 复制"}</button><button className="quiet-button" onClick={() => downloadText("papers.bib", result.bibtex)}>↓ 下载 .bib</button></div></div><pre className="bibtex-panel"><code>{result.bibtex || "% 暂无 BibTeX 引用"}</code></pre></section>}
        </section> : <section className="empty-workspace"><div className="empty-icon"><span>⌕</span></div><h2>准备好开始一项研究</h2><p>输入研究主题并设定检索规模，PaperAgent 会整理论文证据与综述。</p><div className="empty-steps"><span><i>01</i>多源检索</span><span><i>02</i>证据提取</span><span><i>03</i>综述生成</span></div></section>}
      </main>
      <footer className="footer"><span>PaperAgent <i>·</i> 研究证据工作台</span><span>结果由自动化工作流生成，请结合原文复核</span></footer>
    </div>
  );
}

function Metric({ label, value, unit, accent }: { label: string; value: number; unit: string; accent: string }) {
  return <div className={`metric ${accent}`}><span className="metric-label">{label}</span><strong>{value}<small>{unit}</small></strong><span className="metric-rule" /></div>;
}

function downloadText(filename: string, content: string) {
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([content], { type: "text/plain;charset=utf-8" }));
  link.download = filename;
  link.click();
  URL.revokeObjectURL(link.href);
}

function formatDate(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleString("zh-CN", {
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
      });
}

export default App;

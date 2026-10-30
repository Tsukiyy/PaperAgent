"""PaperAgent 领域异常。"""


class PaperAgentError(Exception):
    """所有可预期业务异常的基类。"""


class SearchError(PaperAgentError):
    """论文检索或响应解析失败。"""


class ReviewError(PaperAgentError):
    """指定的综述生成方式执行失败。"""


class SynthesisError(PaperAgentError):
    """基于全文 claims 的横向综合执行失败。"""


class PdfDownloadError(PaperAgentError):
    """公开 PDF 下载或文件校验失败。"""


class PdfParseError(PaperAgentError):
    """PDF 文本、页码或章节解析失败。"""


class EvidenceError(PaperAgentError):
    """带来源结论提取失败。"""

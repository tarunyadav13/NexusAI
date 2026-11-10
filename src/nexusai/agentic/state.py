from typing import TypedDict, List, Dict, Any


class AgenticRAGState(TypedDict, total=False):

    original_query: str

    rewritten_query: str

    current_query: str

    documents: List[Dict[str, Any]]

    web_results: List[Dict[str, Any]]

    use_web: bool

    relevance_score: float

    evidence_sufficient: bool

    retry_count: int

    max_retries: int

    answer: str

    citations: List[Dict[str, Any]]

    error: str
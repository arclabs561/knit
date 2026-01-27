"""Type stubs for rank-fusion Python bindings.

This file provides static type checking support for mypy, pyright, and other type checkers.
"""

from typing import List, Tuple, Optional, Literal

# Type aliases
RankedList = List[Tuple[str, float]]
MultiRankedLists = List[RankedList]
NormalizationType = Literal["zscore", "minmax", "sum", "rank", "none"]


# Configuration Classes
class RrfConfigPy:
    """Configuration for Reciprocal Rank Fusion (RRF)."""
    
    k: int
    top_k: Optional[int]
    
    def __init__(self, k: int = 60) -> None: ...
    def with_top_k(self, top_k: int) -> "RrfConfigPy": ...


class FusionConfigPy:
    """Configuration for general fusion methods."""
    
    top_k: Optional[int]
    
    def __init__(self) -> None: ...
    def with_top_k(self, top_k: int) -> "FusionConfigPy": ...


class WeightedConfigPy:
    """Configuration for weighted fusion."""
    
    weight_a: float
    weight_b: float
    normalize: bool
    top_k: Optional[int]
    
    def __init__(self, weight_a: float, weight_b: float) -> None: ...
    def with_normalize(self, normalize: bool) -> "WeightedConfigPy": ...
    def with_top_k(self, top_k: int) -> "WeightedConfigPy": ...


class StandardizedConfigPy:
    """Configuration for standardized fusion."""
    
    clip_range: Tuple[float, float]
    top_k: Optional[int]
    
    def __init__(self, clip_range: Tuple[float, float] = (-3.0, 3.0)) -> None: ...
    def with_top_k(self, top_k: int) -> "StandardizedConfigPy": ...


class AdditiveMultiTaskConfigPy:
    """Configuration for additive multi-task fusion."""
    
    weights: Tuple[float, float]
    normalization: NormalizationType
    top_k: Optional[int]
    
    def __init__(self, weights: Tuple[float, float] = (1.0, 1.0)) -> None: ...
    def with_normalization(self, normalization: NormalizationType) -> "AdditiveMultiTaskConfigPy": ...
    def with_top_k(self, top_k: int) -> "AdditiveMultiTaskConfigPy": ...


# Explainability Classes
class RetrieverIdPy:
    """Identifier for a retriever in explainability results."""
    
    id: str
    
    def __init__(self, id: str) -> None: ...


class SourceContributionPy:
    """Contribution from a single source retriever."""
    
    retriever_id: RetrieverIdPy
    original_rank: Optional[int]
    original_score: Optional[float]
    contribution: float
    
    def __init__(
        self,
        retriever_id: RetrieverIdPy,
        original_rank: Optional[int],
        original_score: Optional[float],
        contribution: float,
    ) -> None: ...


class ExplanationPy:
    """Explanation of how a fused result was computed."""
    
    sources: List[SourceContributionPy]
    
    def __init__(self, sources: List[SourceContributionPy]) -> None: ...


class FusedResultPy:
    """A fused result with explanation."""
    
    id: str
    score: float
    rank: int
    explanation: ExplanationPy
    
    def __init__(
        self,
        id: str,
        score: float,
        rank: int,
        explanation: ExplanationPy,
    ) -> None: ...


class RetrieverStatsPy:
    """Statistics for a retriever."""
    
    retriever_id: RetrieverIdPy
    total_contributions: float
    avg_contribution: float
    max_contribution: float
    min_contribution: float
    
    def __init__(
        self,
        retriever_id: RetrieverIdPy,
        total_contributions: float,
        avg_contribution: float,
        max_contribution: float,
        min_contribution: float,
    ) -> None: ...


class ConsensusReportPy:
    """Report on consensus across retrievers."""
    
    retriever_stats: List[RetrieverStatsPy]
    consensus_score: float
    
    def __init__(
        self,
        retriever_stats: List[RetrieverStatsPy],
        consensus_score: float,
    ) -> None: ...


# Rank-based Fusion Functions
def rrf(
    results_a: RankedList,
    results_b: RankedList,
    k: int = 60,
    top_k: Optional[int] = None,
) -> RankedList:
    """Reciprocal Rank Fusion (RRF) for two ranked lists."""
    ...


def rrf_multi(
    lists: MultiRankedLists,
    k: int = 60,
    top_k: Optional[int] = None,
) -> RankedList:
    """RRF fusion for multiple ranked lists."""
    ...


def isr(
    results_a: RankedList,
    results_b: RankedList,
    k: int = 1,
    top_k: Optional[int] = None,
) -> RankedList:
    """Inverse Square Rank (ISR) for two ranked lists."""
    ...


def isr_multi(
    lists: MultiRankedLists,
    k: int = 1,
    top_k: Optional[int] = None,
) -> RankedList:
    """ISR fusion for multiple ranked lists."""
    ...


def borda(
    results_a: RankedList,
    results_b: RankedList,
    top_k: Optional[int] = None,
) -> RankedList:
    """Borda count fusion for two ranked lists."""
    ...


def borda_multi(
    lists: MultiRankedLists,
    top_k: Optional[int] = None,
) -> RankedList:
    """Borda count fusion for multiple ranked lists."""
    ...


# Score-based Fusion Functions
def combsum(
    results_a: RankedList,
    results_b: RankedList,
    top_k: Optional[int] = None,
) -> RankedList:
    """CombSUM fusion for two ranked lists."""
    ...


def combsum_multi(
    lists: MultiRankedLists,
    top_k: Optional[int] = None,
) -> RankedList:
    """CombSUM fusion for multiple ranked lists."""
    ...


def combmnz(
    results_a: RankedList,
    results_b: RankedList,
    top_k: Optional[int] = None,
) -> RankedList:
    """CombMNZ fusion for two ranked lists."""
    ...


def combmnz_multi(
    lists: MultiRankedLists,
    top_k: Optional[int] = None,
) -> RankedList:
    """CombMNZ fusion for multiple ranked lists."""
    ...


def weighted(
    results_a: RankedList,
    results_b: RankedList,
    weight_a: float,
    weight_b: float,
    normalize: bool = True,
    top_k: Optional[int] = None,
) -> RankedList:
    """Weighted fusion for two ranked lists."""
    ...


def dbsf(
    results_a: RankedList,
    results_b: RankedList,
    top_k: Optional[int] = None,
) -> RankedList:
    """DBSF (Distribution-Based Score Fusion) for two ranked lists."""
    ...


def dbsf_multi(
    lists: MultiRankedLists,
    top_k: Optional[int] = None,
) -> RankedList:
    """DBSF fusion for multiple ranked lists."""
    ...


def standardized(
    results_a: RankedList,
    results_b: RankedList,
    clip_range: Tuple[float, float] = (-3.0, 3.0),
    top_k: Optional[int] = None,
) -> RankedList:
    """Standardized fusion (ERANK-style) for two ranked lists."""
    ...


def standardized_multi(
    lists: MultiRankedLists,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
    top_k: Optional[int] = None,
) -> RankedList:
    """Standardized fusion for multiple ranked lists."""
    ...


def additive_multi_task(
    results_a: RankedList,
    results_b: RankedList,
    weight_a: Optional[float] = None,
    weight_b: Optional[float] = None,
    normalization: Optional[NormalizationType] = None,
    top_k: Optional[int] = None,
) -> RankedList:
    """Additive multi-task fusion (ResFlow-style) for two ranked lists."""
    ...


# Explainability Functions
def rrf_explain(
    lists: MultiRankedLists,
    retriever_ids: List[RetrieverIdPy],
    k: int = 60,
    top_k: Optional[int] = None,
) -> List[FusedResultPy]:
    """RRF fusion with explainability."""
    ...


def combsum_explain(
    lists: MultiRankedLists,
    retriever_ids: List[RetrieverIdPy],
    top_k: Optional[int] = None,
) -> List[FusedResultPy]:
    """CombSUM fusion with explainability."""
    ...


def combmnz_explain(
    lists: MultiRankedLists,
    retriever_ids: List[RetrieverIdPy],
    top_k: Optional[int] = None,
) -> List[FusedResultPy]:
    """CombMNZ fusion with explainability."""
    ...


def dbsf_explain(
    lists: MultiRankedLists,
    retriever_ids: List[RetrieverIdPy],
    top_k: Optional[int] = None,
) -> List[FusedResultPy]:
    """DBSF fusion with explainability."""
    ...


# Validation Classes
class ValidationResultPy:
    """Result of validation checks on fusion results."""
    
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    
    def __init__(self) -> None: ...


# Validation Functions
def validate_sorted(results: RankedList) -> ValidationResultPy:
    """Validate that fusion results are sorted by score (descending)."""
    ...


def validate_no_duplicates(results: RankedList) -> ValidationResultPy:
    """Validate that fusion results contain no duplicate document IDs."""
    ...


def validate_finite_scores(results: RankedList) -> ValidationResultPy:
    """Validate that all scores are finite (not NaN or Infinity)."""
    ...


def validate_non_negative_scores(results: RankedList) -> ValidationResultPy:
    """Validate that all scores are non-negative (warning only)."""
    ...


def validate_bounds(
    results: RankedList,
    max_results: Optional[int] = None,
) -> ValidationResultPy:
    """Validate that results are within expected bounds."""
    ...


def validate(
    results: RankedList,
    check_non_negative: bool = False,
    max_results: Optional[int] = None,
) -> ValidationResultPy:
    """Comprehensive validation of fusion results.
    
    Performs all validation checks:
    - Sorted by score (descending)
    - No duplicate document IDs
    - All scores are finite
    - Optional: non-negative scores (warning only)
    - Optional: within bounds (warning only)
    """
    ...


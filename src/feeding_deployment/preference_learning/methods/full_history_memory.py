from __future__ import annotations

from typing import Any, Dict, List, Optional

# Character size at which get_memory_block starts warning (~4 chars/token, so
# ~50k tokens). The block is still returned in full -- the "single_full_history"
# baseline is deliberately literal ("store everything, give it all to the LLM");
# the warning just keeps a long deployment from silently creeping toward the
# model's context limit.
_WARN_BLOCK_CHARS = 200_000


class FullHistoryMemoryModel:
    """Verbatim single-layer memory: every prior finalized meal's episode text,
    in chronological order. No summarization, no embeddings, no retrieval --
    prediction gets the complete history. This is the "single_full_history"
    backend of PredictionModel (research baseline against the three-layer
    semantic/episodic/working split)."""

    def __init__(self, max_days: Optional[int] = None) -> None:
        # When set, get_memory_block keeps only the most recent max_days
        # episodes (guard against unbounded prompt growth on long deployments).
        self.max_days = max_days
        self._episode_texts: List[str] = []
        # Structured record per episode ({"day","context","prefs",
        # "corrected_fields"}), aligned with _episode_texts; None for episodes
        # added without metadata. The joint single_full_history path ignores
        # these (it consumes get_memory_block's flat text), but per-dim
        # prediction slices them by field -- same shape EpisodicMemoryModel
        # carries, so dim_prediction._episode_lines reads either identically.
        self._metas: List[Optional[Dict[str, Any]]] = []

    def load_history(
        self,
        episode_texts: List[str],
        metas: Optional[List[Optional[Dict[str, Any]]]] = None,
    ) -> None:
        """Seed the history from persisted prior-day episode texts (chronological),
        optionally with their aligned structured records (for per-dim slicing)."""
        self._episode_texts = list(episode_texts)
        if metas is not None:
            if len(metas) != len(episode_texts):
                raise ValueError(
                    f"metas ({len(metas)}) must align with episode_texts ({len(episode_texts)})"
                )
            self._metas = list(metas)
        else:
            self._metas = [None] * len(episode_texts)

    def add_episode(self, episode_text: str, meta: Optional[Dict[str, Any]] = None) -> None:
        self._episode_texts.append(episode_text)
        self._metas.append(meta)

    def get_memory_block(self) -> str:
        """All stored episodes joined into one prompt block ("" when empty)."""
        texts = self._episode_texts
        if self.max_days is not None and self.max_days >= 0:
            texts = texts[-self.max_days:] if self.max_days else []
        block = "\n\n".join(texts)
        if len(block) > _WARN_BLOCK_CHARS:
            print(
                f"Warning: full-history memory block is {len(block)} characters "
                f"(~{len(block) // 4} tokens) across {len(texts)} episodes; "
                f"consider setting max_days before this hits the model's context limit.",
                flush=True,
            )
        return block

    def get_records(self) -> List[Dict[str, Any]]:
        """Every stored episode as a structured record ({"episode_text","meta"}),
        chronological, honoring the same max_days window as get_memory_block. This
        is the per-dim path's substitute for EpisodicMemoryModel.retrieve_records:
        single_full_history means NO retrieval, so it returns all episodes rather
        than a top-k slice. meta is None for episodes stored without metadata."""
        texts = self._episode_texts
        metas = self._metas
        if self.max_days is not None and self.max_days >= 0:
            if self.max_days:
                texts = texts[-self.max_days:]
                metas = metas[-self.max_days:]
            else:
                texts, metas = [], []
        return [{"episode_text": t, "meta": m} for t, m in zip(texts, metas)]

    def reset(self) -> None:
        self._episode_texts = []
        self._metas = []

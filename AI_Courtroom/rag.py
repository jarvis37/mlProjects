"""
rag.py — Basic Retrieval-Augmented Generation pipeline.

Uses sentence-transformers to embed a small legal corpus and
retrieves the top-k most relevant passages for a given case query.
"""

from __future__ import annotations

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Legal Corpus
# ---------------------------------------------------------------------------

LEGAL_CORPUS: list[str] = [
    "The prosecution must prove guilt beyond a reasonable doubt.",
    "Circumstantial evidence can support a conviction if it is strong and consistent.",
    "Defendants are presumed innocent until proven guilty beyond a reasonable doubt.",
    "Burden of proof lies on the prosecution, not the defense.",
    "Physical evidence found at the scene may establish a direct link to the suspect.",
    "Eyewitness testimony can be compelling but is sometimes unreliable.",
    "Motive, means, and opportunity are three pillars the prosecution may use.",
    "Alibi evidence, if credible, can create reasonable doubt.",
    "Character evidence may be introduced to support or undermine credibility.",
    "The right to remain silent cannot be used as evidence of guilt.",
    "Hearsay evidence is generally inadmissible unless a recognized exception applies.",
    "A chain of custody must be maintained to admit physical evidence.",
    "Flight from the scene may be considered as consciousness of guilt.",
    "Expert witness testimony must be based on reliable methods and sufficient facts.",
    "Reasonable doubt does not mean absolute certainty; it means doubt based on reason.",
]


import logging
logger = logging.getLogger(__name__)

class RAGRetriever:
    """
    Embeds the legal corpus once and provides fast nearest-neighbour retrieval
    via a FAISS flat index (exact cosine similarity over L2-normalised vectors).
    """

    def __init__(
        self,
        corpus: list[str] = LEGAL_CORPUS,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        logger.info("[RAG] Loading embedding model …")
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            logger.error(f"Failed to load sentence transformer model {model_name}: {str(e)}")
            raise

        self.corpus = corpus

        # Embed corpus and build FAISS index
        try:
            embeddings: np.ndarray = self.model.encode(corpus, convert_to_numpy=True)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # L2 normalise

            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner-product == cosine on normalised vecs
            self.index.add(embeddings.astype(np.float32))
            logger.info(f"[RAG] Indexed {len(corpus)} legal passages.")
        except Exception as e:
            logger.error(f"Failed to create embeddings or FAISS index: {str(e)}")
            raise

    def retrieve(self, query: str, top_k: int = 4) -> list[str]:
        """Return the top-k most relevant corpus passages for *query*.
        
        Args:
            query (str): The search text string provided by the user.
            top_k (int): Number of most relevant sentences to extract.
            
        Returns:
            list[str]: list of the extracted strings from the legal corpus.
            
        Raises:
            Exception: Bubbles up any failure during embedding encoding or index search.
        """
        try:
            query_vec: np.ndarray = self.model.encode([query], convert_to_numpy=True)
            query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)

            scores, indices = self.index.search(query_vec.astype(np.float32), top_k)
            results = [self.corpus[i] for i in indices[0] if i < len(self.corpus)]
            return results
        except Exception as e:
            logger.error(f"Failed to retrieve context for query: {str(e)}")
            raise


def retrieve_context(query: str, retriever: RAGRetriever, top_k: int = 4) -> str:
    """
    Convenience wrapper that returns retrieved passages as a numbered string
    suitable for injection into an LLM prompt.
    """
    passages = retriever.retrieve(query, top_k=top_k)
    numbered = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(passages))
    return numbered

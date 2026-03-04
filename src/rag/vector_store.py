"""
H-1B Vector Store
=================
Pinecone-backed semantic search using OpenAI transformer embeddings.

Architecture:
    Company text → OpenAI text-embedding-3-small (1536-dim transformer embedding)
                 → Pinecone serverless index (cosine similarity)
                 → Top-K semantic search results for RAG context
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Real vector store (requires OpenAI + Pinecone API keys)
# ---------------------------------------------------------------------------

class H1BVectorStore:
    """
    Company document store using Pinecone + OpenAI transformer embeddings.

    Each H-1B company is encoded as a natural-language text document and
    embedded using OpenAI's `text-embedding-3-small` transformer model
    (1536 dimensions). Documents are stored in a Pinecone serverless index
    and retrieved via cosine similarity for RAG context injection.

    Usage:
        vs = H1BVectorStore(openai_key="sk-...", pinecone_key="pc-...")
        vs.index_companies(df)
        results = vs.search("tech companies in California", top_k=5)
    """

    INDEX_NAME = "h1b-companies"
    EMBED_MODEL = "text-embedding-3-small"
    DIMENSION   = 1536

    def __init__(self, openai_key: str, pinecone_key: str, index_name: str = INDEX_NAME) -> None:
        from openai import OpenAI
        from pinecone import Pinecone

        self.openai_client = OpenAI(api_key=openai_key)
        self.pc            = Pinecone(api_key=pinecone_key)
        self.index_name    = index_name
        self.index         = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> List[float]:
        """Encode *text* using the OpenAI transformer embedding model."""
        response = self.openai_client.embeddings.create(
            input=text,
            model=self.EMBED_MODEL,
        )
        return response.data[0].embedding

    @staticmethod
    def _row_to_text(row: dict) -> str:
        """Convert a company statistics row to a rich natural-language document."""
        return (
            f"Company: {row['company']}\n"
            f"Headquarters State: {row['state']}\n"
            f"Total H-1B Filings (2024): {row['total_filings']:,}\n"
            f"Average Annual Salary: ${row['avg_salary']:,.0f}\n"
            f"Median Annual Salary: ${row['median_salary']:,.0f}\n"
            f"Sponsorship Score: {row['sponsorship_score']}/100\n"
            f"Company Size: {row['size_category']}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Create the Pinecone index if it does not already exist."""
        import time
        from pinecone import ServerlessSpec

        existing = [idx.name for idx in self.pc.list_indexes()]
        if self.index_name not in existing:
            self.pc.create_index(
                name=self.index_name,
                dimension=self.DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            time.sleep(10)  # Wait for index to become ready

        self.index = self.pc.Index(self.index_name)

    def index_companies(self, df) -> int:
        """
        Embed and upsert all companies in *df* into the Pinecone index.

        Args:
            df: Cleaned H-1B DataFrame.

        Returns:
            Number of vectors upserted.
        """
        if self.index is None:
            self.initialize()

        # Clear stale vectors
        try:
            stats = self.index.describe_index_stats()
            if stats.total_vector_count > 0:
                self.index.delete(delete_all=True, namespace="")
        except Exception:
            pass

        vectors = []
        for idx, row in df.iterrows():
            text      = self._row_to_text(row.to_dict())
            embedding = self._embed(text)
            vectors.append({
                "id":       f"company_{idx}",
                "values":   embedding,
                "metadata": {
                    "company":          str(row["company"]),
                    "state":            str(row["state"]),
                    "total_filings":    int(row["total_filings"]),
                    "avg_salary":       float(row["avg_salary"]),
                    "median_salary":    float(row["median_salary"]),
                    "sponsorship_score":float(row["sponsorship_score"]),
                    "size_category":    str(row["size_category"]),
                },
            })

        batch_size = 50
        for i in range(0, len(vectors), batch_size):
            self.index.upsert(vectors=vectors[i : i + batch_size], namespace="")

        return len(vectors)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Perform semantic search using transformer embeddings + cosine similarity.

        Args:
            query:  Natural-language search query.
            top_k:  Number of top results to return.

        Returns:
            List of dicts with keys: id, score, metadata.
        """
        if self.index is None:
            self.initialize()

        query_vector = self._embed(query)
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            namespace="",
        )
        return [{"id": m.id, "score": m.score, "metadata": m.metadata} for m in results.matches]

    def get_context(self, query: str, top_k: int = 5) -> str:
        """Return retrieved documents formatted as a string for LLM prompt injection."""
        results = self.search(query, top_k)
        if not results:
            return "No relevant companies found."

        lines = ["Relevant H-1B Sponsor Companies (retrieved via semantic search):\n"]
        for i, r in enumerate(results, 1):
            m = r["metadata"]
            lines.append(
                f"{i}. {m['company']} — {int(m['total_filings']):,} filings, "
                f"${float(m['avg_salary']):,.0f} avg salary, "
                f"Score: {m['sponsorship_score']}/100"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Mock vector store (demo mode — no API keys required)
# ---------------------------------------------------------------------------

class MockVectorStore:
    """
    Keyword-based in-memory vector store for demo and testing environments.

    Mimics the H1BVectorStore interface using simple keyword routing
    rather than actual transformer embeddings or Pinecone queries.
    No API keys required.
    """

    def __init__(self, df=None) -> None:
        self.df = df

    def initialize(self) -> None:
        pass

    def index_companies(self, df) -> int:
        self.df = df
        return len(df)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if self.df is None:
            return []

        query_lower = query.lower()
        df = self.df.copy()

        # Exact company name match takes priority
        for _, row in df.iterrows():
            if row["company"].lower() in query_lower:
                return [{"id": "exact_match", "score": 0.95, "metadata": row.to_dict()}]

        # Keyword routing
        if any(w in query_lower for w in ("top", "best", "most", "highest", "leading")):
            df = df.nlargest(top_k, "total_filings")
        elif any(w in query_lower for w in ("salary", "pay", "highest paying", "wage")):
            df = df.nlargest(top_k, "avg_salary")
        elif any(w in query_lower for w in ("tech", "software", "google", "meta", "apple")):
            tech = {"GOOGLE", "META", "APPLE", "AMAZON", "MICROSOFT", "NVIDIA"}
            df = df[df["company"].isin(tech)]
        elif any(w in query_lower for w in ("consult", "infosys", "tcs", "cognizant")):
            consult = {"COGNIZANT", "TCS", "INFOSYS", "DELOITTE", "EY", "ACCENTURE"}
            df = df[df["company"].isin(consult)]
        else:
            df = df.nlargest(top_k, "total_filings")

        return [
            {"id": f"company_{row.name}", "score": 0.85, "metadata": row.to_dict()}
            for _, row in df.head(top_k).iterrows()
        ]

    def get_context(self, query: str, top_k: int = 5) -> str:
        results = self.search(query, top_k)
        if not results:
            return "No relevant companies found."

        lines = ["Relevant H-1B Sponsor Companies:\n"]
        for i, r in enumerate(results, 1):
            m = r["metadata"]
            lines.append(
                f"{i}. {m['company']} — {int(m['total_filings']):,} filings, "
                f"${float(m['avg_salary']):,.0f} avg salary"
            )
        return "\n".join(lines)

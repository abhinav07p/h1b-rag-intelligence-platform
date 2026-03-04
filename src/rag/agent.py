"""
H-1B RAG Agent
==============
GPT-4o Retrieval-Augmented Generation pipeline for domain-specific H-1B Q&A.

Architecture:
    User query → semantic search on Pinecone vector store (transformer embeddings)
               → top-K company context injected into GPT-4o system prompt
               → GPT-4o generates a grounded, cited response
               → RAGResponse returned with answer + source documents
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class RAGResponse:
    """Container for a single RAG pipeline output."""
    response: str
    sources:  List[Dict]
    query:    str


# ---------------------------------------------------------------------------
# Real RAG agent (requires OpenAI API key)
# ---------------------------------------------------------------------------

class H1BRAGAgent:
    """
    RAG agent powered by GPT-4o inference.

    Retrieves relevant company context from the vector store using
    transformer-based semantic similarity, then generates a grounded
    answer via GPT-4o with the retrieved context injected into the prompt.

    Usage:
        agent = H1BRAGAgent(openai_key="sk-...", vector_store=vs)
        response = agent.chat("Which companies in CA pay the most?")
    """

    SYSTEM_PROMPT = (
        "You are an expert H-1B visa sponsorship advisor with access to real "
        "2024 USCIS LCA filing data.\n\n"
        "Help users understand:\n"
        "1. Which companies sponsor H-1B visas\n"
        "2. Sponsorship patterns and salary trends\n"
        "3. How to improve their chances\n\n"
        "Always cite specific data when available. "
        "Include disclaimers about the H-1B lottery (~25% selection rate)."
    )

    def __init__(self, openai_key: str, vector_store) -> None:
        from openai import OpenAI
        self.client       = OpenAI(api_key=openai_key)
        self.vector_store = vector_store
        self.history: List[Dict] = []

    def chat(self, user_message: str, top_k: int = 5) -> RAGResponse:
        """
        Run a single RAG query-response cycle.

        Steps:
            1. Retrieve top-K semantically similar company docs from vector store.
            2. Format retrieved context as a system message.
            3. Append conversation history (last 6 turns) for multi-turn coherence.
            4. Call GPT-4o and return the response.

        Args:
            user_message: The user's natural-language question.
            top_k:        Number of company documents to retrieve.

        Returns:
            RAGResponse with the generated answer and source documents.
        """
        context = self.vector_store.get_context(user_message, top_k)
        sources = self.vector_store.search(user_message, top_k)

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "system", "content": f"RETRIEVED DATA:\n{context}"},
            *self.history[-6:],
            {"role": "user",   "content": user_message},
        ]

        completion = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=800,
        )
        answer = completion.choices[0].message.content

        self.history.extend([
            {"role": "user",      "content": user_message},
            {"role": "assistant", "content": answer},
        ])

        return RAGResponse(response=answer, sources=sources, query=user_message)

    def clear_history(self) -> None:
        """Reset the conversation history."""
        self.history = []


# ---------------------------------------------------------------------------
# Mock RAG agent (demo mode — no API keys required)
# ---------------------------------------------------------------------------

class MockRAGAgent:
    """
    Keyword-routing RAG agent for demo and testing environments.

    Mimics the H1BRAGAgent interface but returns template responses based
    on simple keyword matching rather than actual GPT-4o inference.
    No API keys required.
    """

    def __init__(self, vector_store, df=None) -> None:
        self.vector_store = vector_store
        self.df = df

    def chat(self, user_message: str, top_k: int = 5) -> RAGResponse:
        sources    = self.vector_store.search(user_message, top_k)
        query      = user_message.lower()

        if any(w in query for w in ("top", "best", "most", "leading")):
            response = self._top_sponsors(sources)
        elif any(w in query for w in ("salary", "pay", "money", "wage")):
            response = self._salary_info(sources)
        elif any(w in query for w in ("lottery", "chance", "odds", "rate")):
            response = self._lottery_info()
        elif any(w in query for w in ("tip", "advice", "strategy", "improve")):
            response = self._advice()
        else:
            response = self._general(sources)

        return RAGResponse(response=response, sources=sources, query=user_message)

    # ------------------------------------------------------------------
    # Template responses
    # ------------------------------------------------------------------

    def _top_sponsors(self, sources: List[Dict]) -> str:
        if not sources:
            return "Data not available."
        lines = "\n".join(
            f"• **{s['metadata']['company']}**: "
            f"{int(s['metadata']['total_filings']):,} filings, "
            f"${float(s['metadata']['avg_salary']):,.0f} avg salary"
            for s in sources[:5]
        )
        return (
            f"**Top H-1B Sponsors (2024 Real Data):**\n\n{lines}\n\n"
            "**Key Insights:**\n"
            "- Amazon leads with highest filing volume\n"
            "- Tech companies offer higher salaries ($170K–$210K)\n"
            "- Consulting firms have high volume but lower salaries\n\n"
            "⚠️ **Remember:** H-1B is a lottery system with ~25% selection rate."
        )

    def _salary_info(self, sources: List[Dict]) -> str:
        return (
            "**H-1B Salary Ranges (2024):**\n\n"
            "• Big Tech (Meta, Google, Apple): $170,000 – $230,000\n"
            "• Finance (JPMorgan, Goldman): $150,000 – $180,000\n"
            "• Enterprise Software (Salesforce): $150,000 – $175,000\n"
            "• IT Consulting (Cognizant, TCS): $90,000 – $115,000\n\n"
            "H-1B requires \"prevailing wage\" — companies must meet market rate."
        )

    def _lottery_info(self) -> str:
        return (
            "**H-1B Lottery System:**\n\n"
            "• Regular cap: 65,000 visas\n"
            "• Master's exemption: +20,000\n"
            "• Selection rate: ~25–30%\n\n"
            "**Timeline:**\n"
            "- March: Registration\n"
            "- Late March: Lottery results\n"
            "- October 1: Employment start date\n\n"
            "**Cap-Exempt:** Universities and nonprofit research orgs (no lottery)"
        )

    def _advice(self) -> str:
        return (
            "**H-1B Strategy Tips:**\n\n"
            "1. Target companies with 500+ annual filings\n"
            "2. Apply to 10–20 sponsors minimum\n"
            "3. Master's degree = extra lottery pool\n"
            "4. Have backup plans (STEM OPT, O-1, EB-1)\n"
            "5. Start applications 6–12 months early\n\n"
            "**Avoid:** Companies that ask you to pay for sponsorship costs"
        )

    def _general(self, sources: List[Dict]) -> str:
        if sources:
            top = sources[0]["metadata"]
            return (
                f"Based on 2024 USCIS data:\n\n"
                f"**{top['company']}**: {int(top['total_filings']):,} filings, "
                f"${float(top['avg_salary']):,.0f} avg salary\n\n"
                "I can help with:\n"
                "- Top sponsors by volume or salary\n"
                "- Salary comparisons across industries\n"
                "- H-1B lottery process and timeline\n"
                "- Job search and application strategy\n\n"
                "What would you like to know?"
            )
        return "How can I help you with H-1B sponsorship questions?"

    def clear_history(self) -> None:
        pass

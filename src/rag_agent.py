"""
H-1B RAG Agent
==============
GPT-4o RAG pipeline for H-1B Q&A.
"""

from typing import List, Dict
from dataclasses import dataclass


@dataclass
class RAGResponse:
    response: str
    sources: List[Dict]
    query: str


class H1BRAGAgent:
    """RAG agent using GPT-4o."""
    
    SYSTEM_PROMPT = """You are an expert H-1B visa sponsorship advisor with access to real 2024 USCIS filing data.

Help users understand:
1. Which companies sponsor H-1B visas
2. Sponsorship patterns and salary trends
3. How to improve their chances

Always cite specific data when available. Include disclaimers about the H-1B lottery (~25% selection rate)."""

    def __init__(self, openai_key: str, vector_store):
        from openai import OpenAI
        self.client = OpenAI(api_key=openai_key)
        self.vector_store = vector_store
        self.history = []
    
    def chat(self, user_message: str, top_k: int = 5) -> RAGResponse:
        context = self.vector_store.get_context(user_message, top_k)
        sources = self.vector_store.search(user_message, top_k)
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "system", "content": f"DATA:\n{context}"},
        ]
        
        for msg in self.history[-6:]:
            messages.append(msg)
        
        messages.append({"role": "user", "content": user_message})
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=800
        )
        
        answer = response.choices[0].message.content
        
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": answer})
        
        return RAGResponse(response=answer, sources=sources, query=user_message)
    
    def clear_history(self):
        self.history = []


class MockRAGAgent:
    """Mock RAG agent for demo mode."""
    
    def __init__(self, vector_store, df=None):
        self.vector_store = vector_store
        self.df = df
    
    def chat(self, user_message: str, top_k: int = 5) -> RAGResponse:
        sources = self.vector_store.search(user_message, top_k)
        query_lower = user_message.lower()
        
        if any(w in query_lower for w in ['top', 'best', 'most', 'leading']):
            response = self._top_sponsors(sources)
        elif any(w in query_lower for w in ['salary', 'pay', 'money']):
            response = self._salary_info(sources)
        elif any(w in query_lower for w in ['lottery', 'chance', 'odds']):
            response = self._lottery_info()
        elif any(w in query_lower for w in ['tip', 'advice', 'strategy']):
            response = self._advice()
        else:
            response = self._general(sources)
        
        return RAGResponse(response=response, sources=sources, query=user_message)
    
    def _top_sponsors(self, sources):
        if not sources:
            return "Data not available."
        
        companies = "\n".join([
            f"• **{s['metadata']['company']}**: {int(s['metadata']['total_filings']):,} filings, ${float(s['metadata']['avg_salary']):,.0f}"
            for s in sources[:5]
        ])
        
        return f"""**Top H-1B Sponsors (2024 Real Data):**

{companies}

**Key Insights:**
- Amazon leads with highest filing volume
- Tech companies offer higher salaries ($170K-$210K)
- Consulting firms have high volume but lower salaries

⚠️ **Remember:** H-1B is a lottery system with ~25% selection rate."""

    def _salary_info(self, sources):
        return """**H-1B Salary Ranges (2024):**

• Big Tech (Meta, Google, Apple): $170,000 - $230,000
• Finance (JPMorgan, Goldman): $150,000 - $180,000
• Enterprise Software (Salesforce): $150,000 - $175,000
• IT Consulting (Cognizant, TCS): $90,000 - $115,000

H-1B requires "prevailing wage" - companies must meet market rate."""

    def _lottery_info(self):
        return """**H-1B Lottery System:**

• Regular cap: 65,000 visas
• Master's exemption: +20,000
• Selection rate: ~25-30%

**Timeline:**
- March: Registration
- Late March: Lottery results
- October 1: Start date

**Cap-Exempt:** Universities, nonprofit research (no lottery)"""

    def _advice(self):
        return """**H-1B Strategy Tips:**

1. Target companies with 500+ annual filings
2. Apply to 10-20 sponsors minimum
3. Master's degree = extra lottery pool
4. Have backup plans (STEM OPT, O-1)
5. Start applications 6-12 months early

**Avoid:** Companies asking you to pay for sponsorship"""

    def _general(self, sources):
        if sources:
            top = sources[0]['metadata']
            return f"""Based on 2024 data:

**{top['company']}**: {int(top['total_filings']):,} filings, ${float(top['avg_salary']):,.0f} avg salary

I can help with:
- Top sponsors by industry
- Salary comparisons
- Lottery process
- Job search strategy

What would you like to know?"""
        
        return "How can I help you with H-1B sponsorship questions?"

    def clear_history(self):
        pass

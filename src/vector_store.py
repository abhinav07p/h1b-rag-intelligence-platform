"""
H-1B Vector Store
=================
Pinecone vector database integration for semantic search.
"""

from typing import List, Dict, Optional
import numpy as np


class H1BVectorStore:
    """Vector store using Pinecone + OpenAI embeddings."""
    
    def __init__(self, openai_key: str, pinecone_key: str, index_name: str = "h1b-companies"):
        from openai import OpenAI
        from pinecone import Pinecone
        
        self.openai_client = OpenAI(api_key=openai_key)
        self.pc = Pinecone(api_key=pinecone_key)
        self.index_name = index_name
        self.index = None
        self.dimension = 1536
    
    def _get_embedding(self, text: str) -> List[float]:
        response = self.openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    
    def _company_to_text(self, row: dict) -> str:
        return f"""Company: {row['company']}
Location: {row['state']}
Total H-1B Filings (2024): {row['total_filings']:,}
Average Salary: ${row['avg_salary']:,.0f}
Median Salary: ${row['median_salary']:,.0f}
Sponsorship Score: {row['sponsorship_score']}/100
Company Size: {row['size_category']}"""
    
    def initialize(self):
        from pinecone import ServerlessSpec
        import time
        
        # List existing indexes
        existing_indexes = self.pc.list_indexes()
        existing_names = [idx.name for idx in existing_indexes]
        
        if self.index_name not in existing_names:
            print(f"Creating index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            # Wait for index to be ready
            time.sleep(10)
        
        self.index = self.pc.Index(self.index_name)
    
    def index_companies(self, df) -> int:
        if self.index is None:
            self.initialize()
        
        # Clear existing data - skip if empty
        try:
            stats = self.index.describe_index_stats()
            if stats.total_vector_count > 0:
                self.index.delete(delete_all=True, namespace="")
        except Exception as e:
            print(f"Note: {e}")
        
        vectors = []
        for idx, row in df.iterrows():
            text = self._company_to_text(row.to_dict())
            embedding = self._get_embedding(text)
            
            vectors.append({
                'id': f"company_{idx}",
                'values': embedding,
                'metadata': {
                    'company': str(row['company']),
                    'state': str(row['state']),
                    'total_filings': int(row['total_filings']),
                    'avg_salary': float(row['avg_salary']),
                    'median_salary': float(row['median_salary']),
                    'sponsorship_score': float(row['sponsorship_score']),
                    'size_category': str(row['size_category']),
                }
            })
        
        # Upsert in batches
        batch_size = 50
        for i in range(0, len(vectors), batch_size):
            self.index.upsert(vectors=vectors[i:i+batch_size], namespace="")
        
        return len(vectors)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if self.index is None:
            self.initialize()
        
        query_embedding = self._get_embedding(query)
        
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=""
        )
        
        return [{'id': m.id, 'score': m.score, 'metadata': m.metadata} for m in results.matches]
    
    def get_context(self, query: str, top_k: int = 5) -> str:
        results = self.search(query, top_k)
        
        if not results:
            return "No relevant companies found."
        
        lines = ["Relevant H-1B Sponsor Companies:\n"]
        for i, r in enumerate(results, 1):
            m = r['metadata']
            lines.append(f"{i}. {m['company']} - {m['total_filings']:,} filings, ${m['avg_salary']:,.0f} avg salary")
        
        return "\n".join(lines)


class MockVectorStore:
    """Mock vector store for demo mode."""
    
    def __init__(self, df=None):
        self.df = df
    
    def initialize(self):
        pass
    
    def index_companies(self, df) -> int:
        self.df = df
        return len(df)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if self.df is None:
            return []
        
        query_lower = query.lower()
        df = self.df.copy()
        
        for _, row in df.iterrows():
            if row['company'].lower() in query_lower:
                return [{'id': 'match', 'score': 0.95, 'metadata': row.to_dict()}]
        
        if any(w in query_lower for w in ['top', 'best', 'most', 'highest', 'leading']):
            df = df.nlargest(top_k, 'total_filings')
        elif any(w in query_lower for w in ['salary', 'pay', 'highest paying']):
            df = df.nlargest(top_k, 'avg_salary')
        elif any(w in query_lower for w in ['tech', 'software', 'google', 'meta', 'apple']):
            tech = ['GOOGLE', 'META', 'APPLE', 'AMAZON', 'MICROSOFT', 'NVIDIA']
            df = df[df['company'].isin(tech)]
        elif any(w in query_lower for w in ['consult', 'infosys', 'tcs', 'cognizant']):
            consult = ['COGNIZANT', 'TCS', 'INFOSYS', 'DELOITTE', 'EY', 'ACCENTURE']
            df = df[df['company'].isin(consult)]
        else:
            df = df.nlargest(top_k, 'total_filings')
        
        results = []
        for _, row in df.head(top_k).iterrows():
            results.append({'id': f"company_{row.name}", 'score': 0.85, 'metadata': row.to_dict()})
        
        return results
    
    def get_context(self, query: str, top_k: int = 5) -> str:
        results = self.search(query, top_k)
        
        if not results:
            return "No relevant companies found."
        
        lines = ["Relevant H-1B Sponsor Companies:\n"]
        for i, r in enumerate(results, 1):
            m = r['metadata']
            lines.append(f"{i}. {m['company']} - {int(m['total_filings']):,} filings, ${float(m['avg_salary']):,.0f}")
        
        return "\n".join(lines)

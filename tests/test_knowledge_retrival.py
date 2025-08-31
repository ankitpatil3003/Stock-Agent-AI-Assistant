from app.agents.knowledge import KnowledgeAgent
from app.config.settings import settings

def test_search_empty_query_returns_empty():
    agent = KnowledgeAgent(settings=settings)
    hits = agent.search_strategies("", k=3)
    assert hits == []

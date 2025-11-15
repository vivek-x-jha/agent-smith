"""Search and scraping helpers used by researcher agents."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

from ..logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """Normalized representation of search hits."""

    title: str
    url: str
    snippet: str
    source: str

    def asdict(self) -> dict[str, str]:
        return asdict(self)


def duckduckgo_search(query: str, max_results: int = 5) -> list[SearchResult]:
    """Perform a web search via DuckDuckGo (no API key required)."""

    logger.info("duckduckgo_search", query=query, max_results=max_results)
    hits: list[SearchResult] = []
    with DDGS() as ddgs:
        for item in ddgs.text(query, max_results=max_results):
            hits.append(
                SearchResult(
                    title=item.get("title") or "Untitled result",
                    url=item.get("href") or item.get("url") or "",
                    snippet=item.get("body") or item.get("description") or "",
                    source="duckduckgo",
                )
            )
    return hits


def wikipedia_search(topic: str, sentences: int = 3) -> list[SearchResult]:
    """Query the public Wikipedia REST API for concise topic summaries."""

    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}".replace(" ", "%20")
    logger.info("wikipedia_search", topic=topic)
    with httpx.Client(timeout=10.0) as client:
        response = client.get(url)
        response.raise_for_status()
    payload = response.json()
    extract = payload.get("extract") or ""
    snippet = " ".join(extract.split()[: sentences * 20])  # approx sentences
    title = payload.get("title") or topic
    canonical_url = payload.get("content_urls", {}).get("desktop", {}).get("page", url)
    return [SearchResult(title=title, url=canonical_url, snippet=snippet, source="wikipedia")]


def arxiv_search(query: str, max_results: int = 3) -> list[SearchResult]:
    """Lightweight arXiv API wrapper that parses Atom feeds."""

    base_url = "http://export.arxiv.org/api/query"
    params = {"search_query": f"all:{query}", "start": 0, "max_results": max_results}
    logger.info("arxiv_search", query=query, max_results=max_results)
    with httpx.Client(timeout=10.0) as client:
        response = client.get(base_url, params=params)
        response.raise_for_status()
    soup = BeautifulSoup(response.text, "xml")
    hits: list[SearchResult] = []
    for entry in soup.find_all("entry"):
        title = (entry.title or "Untitled").text.strip()
        summary = (entry.summary or "").text.strip().replace("\n", " ")
        link_tag = entry.find("link", attrs={"title": "pdf"}) or entry.find("link", attrs={"rel": "alternate"})
        href = link_tag["href"] if link_tag and link_tag.has_attr("href") else ""
        hits.append(SearchResult(title=title, url=href, snippet=summary, source="arxiv"))
    return hits


def to_serializable(results: Iterable[SearchResult]) -> list[dict[str, str]]:
    """Helper to convert data classes into JSON-friendly dictionaries."""

    return [result.asdict() for result in results]


__all__ = [
    "SearchResult",
    "arxiv_search",
    "duckduckgo_search",
    "to_serializable",
    "wikipedia_search",
]

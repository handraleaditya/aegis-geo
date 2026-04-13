"""Content fetching, parsing, and boilerplate stripping.

Handles both URL fetching (via httpx) and raw HTML/text input.
Provides utilities for extracting structured elements (paragraphs, headings)
and cleaning content for NLP analysis.
"""

from __future__ import annotations

import copy
import re

import httpx
from bs4 import BeautifulSoup

# HTML tags considered boilerplate — removed before readability analysis
_BOILERPLATE_TAGS = {"nav", "footer", "header", "aside", "script", "style", "noscript"}

FETCH_TIMEOUT = 10.0  # seconds


async def fetch_url(url: str) -> str:
    """Fetch raw HTML from a URL.

    Uses httpx with redirect-following and a 10-second timeout.
    Raises httpx exceptions on failure for the caller to handle.
    """
    async with httpx.AsyncClient(follow_redirects=True, timeout=FETCH_TIMEOUT) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.text


def parse_html(raw_html: str) -> BeautifulSoup:
    """Parse raw HTML string into a BeautifulSoup tree."""
    return BeautifulSoup(raw_html, "html.parser")


def strip_boilerplate(soup: BeautifulSoup) -> BeautifulSoup:
    """Remove boilerplate elements (nav, footer, header, etc.) in-place.

    Decomposes tags that typically contain non-content elements like
    navigation, footers, and scripts to improve readability scoring accuracy.
    """
    for tag in soup.find_all(True):
        if tag.name in _BOILERPLATE_TAGS:
            tag.decompose()
    return soup


def get_body_text(soup: BeautifulSoup) -> str:
    """Extract cleaned visible text after stripping boilerplate.

    Creates a deep copy to avoid mutating the original soup tree,
    since strip_boilerplate modifies the tree in place.
    """
    stripped = strip_boilerplate(copy.deepcopy(soup))
    return stripped.get_text(separator=" ", strip=True)


def get_first_paragraph(raw_html: str, plain_text: str | None = None) -> str:
    """Extract the first meaningful paragraph from content.

    Strategy:
      1. Try to find the first <p> tag with non-empty text (HTML input).
      2. Fall back to splitting on double newlines (plain text input).
      3. Fall back to the full text if no paragraph breaks exist.
    """
    soup = parse_html(raw_html)
    p_tag = soup.find("p")
    if p_tag and p_tag.get_text(strip=True):
        return p_tag.get_text(strip=True)

    # Fallback: treat as plain text
    text = plain_text or raw_html
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs[0] if paragraphs else text.strip()


def get_heading_tags(soup: BeautifulSoup) -> list[dict[str, str]]:
    """Extract all heading tags (h1-h6) in DOM order.

    Returns a list of dicts with 'tag' (e.g. 'h1') and 'text' keys,
    preserving the order headings appear in the document.
    """
    return [
        {"tag": tag.name, "text": tag.get_text(strip=True)}
        for tag in soup.find_all(re.compile(r"^h[1-6]$"))
    ]

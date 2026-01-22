#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""search_web.py

EN: Tavily-backed web search tool for pwf_mvp_v4_5.py.
ZH: 基于 Tavily API 的 Web 搜索工具，供 pwf_mvp_v4_5.py 调用。

CLI (flags):
  python3 search_web.py --query "..." --topk 5

Env:
  TAVILY_API_KEY: Tavily API key (e.g., tvly-...).

Output:
  - On success: prints up to topk lines, each a compact string for one result.
  - If no results: prints 'no web search result'
  - On error: prints 'no web search result' to stdout and details to stderr, exits non-zero.
"""

import os
import sys
import json
import argparse
from urllib import request
from urllib.error import HTTPError, URLError

TAVILY_SEARCH_URL = "https://api.tavily.com/search"  # Tavily base URL + /search


def _get_api_key():
    # Prefer standard env var name.
    return os.getenv("TAVILY_API_KEY", "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").strip()


def _post_json(url, headers, payload, timeout=30):
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, method="POST")
    for k, v in headers.items():
        req.add_header(k, v)
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    return body


def _format_results(results, topk):
    out_lines = []
    for i, r in enumerate(results[:topk], 1):
        title = (r.get("title") or "").strip()
        url = (r.get("url") or "").strip()
        content = (r.get("content") or "").strip()
        # Keep each line compact; pwf_mvp_v4_5 will additionally truncate when persisting.
        if len(content) > 300:
            content = content[:300] + "..."
        parts = []
        if title:
            parts.append(title)
        if url:
            parts.append(url)
        if content:
            parts.append(content)
        line = " | ".join(parts).strip()
        if not line:
            continue
        out_lines.append(f"{i}. {line}")
    return out_lines


def main():
    parser = argparse.ArgumentParser(description="Tavily web search tool")
    parser.add_argument("--query", required=True, help="Search query string")
    parser.add_argument("--topk", type=int, default=5, help="Number of top results to return (0-20)")
    parser.add_argument("--search_depth", default="basic", choices=["basic", "advanced", "fast", "ultra-fast"],
                        help="Tavily search depth")
    args = parser.parse_args()

    query = (args.query or "").strip()
    topk = int(args.topk)
    if topk < 0:
        topk = 0
    if topk > 20:
        topk = 20

    api_key = _get_api_key()
    if not api_key:
        print("no web search result")
        print("Missing env TAVILY_API_KEY", file=sys.stderr)
        sys.exit(2)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "query": query,
        "max_results": topk if topk else 1,
        "search_depth": args.search_depth,
        "include_answer": False,
        "include_raw_content": False,
    }

    try:
        body = _post_json(TAVILY_SEARCH_URL, headers, payload, timeout=30)
        obj = json.loads(body)
        results = obj.get("results") or []
        if not results:
            print("no web search result")
            return
        lines = _format_results(results, topk if topk else 1)
        if not lines:
            print("no web search result")
            return
        print("\n".join(lines))
    except HTTPError as e:
        # Print a safe fallback to stdout; details to stderr.
        print("no web search result")
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = ""
        print(f"HTTPError: {e.code} {e.reason}", file=sys.stderr)
        if err_body:
            print(err_body[:500], file=sys.stderr)
        sys.exit(1)
    except URLError as e:
        print("no web search result")
        print(f"URLError: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print("no web search result")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

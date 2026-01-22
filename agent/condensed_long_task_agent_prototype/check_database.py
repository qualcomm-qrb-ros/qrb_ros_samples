#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_database.py

EN: Placeholder local database search tool.
ZH: 本地数据库搜索占位工具（当前无可用数据）。

CLI (flags):
  python3 check_database.py --query "..." --topk 5

Output:
  Always prints: 'no available data in local database'
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Local database search tool (placeholder)")
    parser.add_argument("--query", required=True, help="Search query string")
    parser.add_argument("--topk", type=int, default=5, help="Number of top results to return")
    _ = parser.parse_args()

    print("no available data in local database")


if __name__ == "__main__":
    main()

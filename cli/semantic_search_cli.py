#!/usr/bin/env python3

import argparse
from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("verify", help="Verify model loads correctly")

    subparsers.add_parser("verify_embeddings", help="Build or load movie embeddings")

    embed_parser = subparsers.add_parser("embed_text", help="Embed text")
    embed_parser.add_argument("text", type=str, help="Text to embed")

    query_parser = subparsers.add_parser("embed_query", help="Embed a search query")
    query_parser.add_argument("query", type=str, help="Query text")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)

        case "verify_embeddings":
            verify_embeddings()

        case "embed_query":
            embed_query_text(args.query)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

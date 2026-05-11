import argparse
import os

from dotenv import load_dotenv
from google import genai
from lib.hybrid_search import (
    normalize_scores,
    weighted_search_command,
    rrf_search_command,
)


def get_gemini_client():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    return genai.Client(api_key=api_key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    load_dotenv()

    api_key = os.environ.get("GEMINI_API_KEY")
    client = None

    if api_key:
        client = genai.Client(api_key=api_key)

    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalize a list of scores"
    )
    normalize_parser.add_argument(
        "scores", nargs="+", type=float, help="List of scores to normalize"
    )

    weighted_parser = subparsers.add_parser(
        "weighted-search", help="Perform weighted hybrid search"
    )
    weighted_parser.add_argument("query", type=str, help="Search query")
    weighted_parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for BM25 vs semantic (0=all semantic, 1=all BM25, default=0.5)",
    )
    weighted_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return (default=5)"
    )

    rrf_parser = subparsers.add_parser(
        "rrf-search",
        help="Perform Reciprocal Rank Fusion search",
    )

    rrf_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite"],
        help="Query enhancement method",
    )

    rrf_parser.add_argument("query", type=str, help="Search query")

    rrf_parser.add_argument(
        "-k",
        type=int,
        default=60,
        help="RRF k parameter (default=60)",
    )

    rrf_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to return",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized = normalize_scores(args.scores)
            for score in normalized:
                print(f"* {score:.4f}")
        case "weighted-search":
            result = weighted_search_command(args.query, args.alpha, args.limit)

            print(
                f"Weighted Hybrid Search Results for '{result['query']}' (alpha={result['alpha']}):"
            )
            print(
                f"  Alpha {result['alpha']}: {int(result['alpha'] * 100)}% Keyword, {int((1 - result['alpha']) * 100)}% Semantic"
            )
            for i, res in enumerate(result["results"], 1):
                print(f"{i}. {res['title']}")
                print(f"   Hybrid Score: {res.get('score', 0):.3f}")
                metadata = res.get("metadata", {})
                if "bm25_score" in metadata and "semantic_score" in metadata:
                    print(
                        f"   BM25: {metadata['bm25_score']:.3f}, Semantic: {metadata['semantic_score']:.3f}"
                    )
                print(f"   {res['document'][:100]}...")
                print()

        case "rrf-search":
            query = args.query

            if args.enhance in ("spell", "rewrite"):
                client = get_gemini_client()

                if args.enhance == "spell":
                    prompt = f"""Fix any spelling errors in the user-provided movie search query below.
        Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
        Preserve punctuation and capitalization unless needed.
        If unsure, output original unchanged.
        Output only the final query text.

        User query: "{query}"
        """

                elif args.enhance == "rewrite":
                    prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

        Consider:
        - Common movie knowledge (famous actors, popular films)
        - Genre conventions
        - Keep it under 10 words
        - Google-style search query
        - No boolean logic

        Examples:
        - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
        - "movie about bear in london with marmalade" -> "Paddington London marmalade"
        - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

        If you cannot improve it, output original unchanged.
        Output only the rewritten query text.

        User query: "{query}"
        """

                response = client.models.generate_content(
                    model="gemma-4-31b-it",
                    contents=prompt,
                )

                enhanced_query = response.text.strip()

                print(
                    f"Enhanced query ({args.enhance}): '{query}' -> '{enhanced_query}'\n"
                )

                query = enhanced_query

            result = rrf_search_command(query, args.k, args.limit)

            print(f"RRF Search Results for '{query}':\n")

            for i, res in enumerate(result["results"], 1):
                print(f"{i}. {res['title']}")
                print(f"   RRF Score: {res['score']:.3f}")

                metadata = res.get("metadata", {})
                print(
                    f"   BM25 Rank: {metadata.get('bm25_rank', '-')}, "
                    f"Semantic Rank: {metadata.get('semantic_rank', '-')}"
                )

                print(f"   {res['document'][:100]}...\n")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

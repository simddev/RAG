import string
import argparse
import json
from nltk.stem import PorterStemmer

translator = str.maketrans("", "", string.punctuation)
stemmer = PorterStemmer()


def tokenize(text: str) -> list[str]:
    return [token for token in text.lower().translate(translator).split() if token]


def remove_stopwords(tokens: list[str], stopwords: set[str]) -> list[str]:
    return [t for t in tokens if t not in stopwords]


def stem_tokens(tokens: list[str]) -> list[str]:
    return [stemmer.stem(t) for t in tokens]


def matches(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for q in query_tokens:
        for t in title_tokens:
            if q in t:
                return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    with open("data/stopwords.txt", "r", encoding="utf-8") as f:
        stopwords = set(f.read().splitlines())

    match args.command:
        case "search":
            query_tokens = tokenize(args.query)
            query_tokens = remove_stopwords(query_tokens, stopwords)
            query_tokens = stem_tokens(query_tokens)

            with open("data/movies.json", "r", encoding="utf-8") as f:
                data = json.load(f)

            results = []

            for movie in data["movies"]:
                title_tokens = tokenize(movie["title"])
                title_tokens = remove_stopwords(title_tokens, stopwords)
                title_tokens = stem_tokens(title_tokens)

                if matches(query_tokens, title_tokens):
                    results.append(movie)

            results = results[:5]

            print(f"Searching for: {args.query}")
            for i, movie in enumerate(results, start=1):
                print(f"{i}. {movie['title']}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

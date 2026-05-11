import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalize a list of scores"
    )

    normalize_parser.add_argument("scores", nargs="*", type=float)

    args = parser.parse_args()

    match args.command:

        case "normalize":
            scores = args.scores

            if not scores:
                return

            min_score = min(scores)
            max_score = max(scores)

            if min_score == max_score:
                for _ in scores:
                    print("* 1.0000")
                return

            normalized = [(s - min_score) / (max_score - min_score) for s in scores]

            for s in normalized:
                print(f"* {s:.4f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

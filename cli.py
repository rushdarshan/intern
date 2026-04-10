import argparse
import logging

from graph import run_question


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def _print_result(result: dict, show_metadata: bool) -> None:
    answer = result.get("answer", "I don't know based on the available context.")
    print(f"\nAssistant: {answer}")

    if not show_metadata:
        return

    route = result.get("route", "unknown")
    route_reason = result.get("route_reason", "")
    warnings = result.get("warnings", [])
    sources = result.get("sources", [])

    print(f"Route: {route}")
    if route_reason:
        print(f"Reason: {route_reason}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    if sources:
        print("Sources:")
        for index, source in enumerate(sources, start=1):
            title = source.get("title") or source.get("url") or "Untitled source"
            url = source.get("url", "")
            print(f"  [{index}] {title} - {url}")


def _interactive_loop(show_metadata: bool) -> None:
    print("Debales AI Assistant CLI")
    print("Type `exit` or `quit` to stop.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Exiting.")
            return

        _print_result(run_question(question), show_metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description="Debales AI Assistant CLI")
    parser.add_argument("question", nargs="*", help="Optional one-shot question.")
    parser.add_argument(
        "--show-metadata",
        action="store_true",
        help="Print route, warnings, and source URLs after each answer.",
    )
    args = parser.parse_args()

    if args.question:
        _print_result(run_question(" ".join(args.question)), args.show_metadata)
        return

    _interactive_loop(args.show_metadata)


if __name__ == "__main__":
    main()

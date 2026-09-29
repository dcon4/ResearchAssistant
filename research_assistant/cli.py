import argparse
import sys

from research_assistant import logger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="research-assistant")
    sub = parser.add_subparsers(dest="command")
    serve = sub.add_parser("serve", help="start the web page")
    serve.add_argument("--port", type=int, default=None)
    sub.add_parser("log", help="print this launch's log file path")
    sub.add_parser("index", help="scan folders and (re)build the index")
    sub.add_parser("reembed", help="rebuild all vectors (after mode change)")
    ask_cmd = sub.add_parser("ask", help="ask a question from the terminal")
    ask_cmd.add_argument("question", nargs="+")
    args = parser.parse_args(argv)

    logger.open_log()
    if args.command == "log":
        print(logger.current_log_path())
        return 0
    if args.command == "index":
        from research_assistant.pipeline import rescan

        result = rescan()
        print(result)
        return 0
    if args.command == "reembed":
        from research_assistant.pipeline import reembed

        print(reembed())
        return 0
    if args.command == "ask":
        from research_assistant.answer import ask

        result = ask(" ".join(args.question))
        if result["answer"]:
            print(result["answer"])
        else:
            print(result["error"])
        for index, source in enumerate(result["sources"], start=1):
            print(f"[{index}] {source.file_path} ({source.location})")
        return 0
    if args.command == "serve":
        from research_assistant.config import load_settings, save_settings
        from research_assistant.webui import serve

        if args.port:
            settings = load_settings()
            settings["port"] = args.port
            save_settings(settings)
        serve()
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

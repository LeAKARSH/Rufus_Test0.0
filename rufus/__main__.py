"""Console entry point: ``python -m rufus [subcommand]``.

``python -m rufus`` (nothing else) opens the interactive
:mod:`rufus.shell` REPL. Explicit subcommands:

- ``python -m rufus daemon``  — the 24/7 scheduler (previously the default)
- ``python -m rufus report``  — render the daily report
- ``python -m rufus paper``   — paper-trading summary
- ``python -m rufus ticker``  — manage the watchlist
"""

import sys


def main() -> None:
    argv = sys.argv[1:]
    index = next(
        (i for i, a in enumerate(argv) if not a.startswith("-")), None
    )
    if index is not None:
        sub = argv[index]
        rest = argv[index + 1:]
    else:
        sub, rest = None, argv
    if sub == "daemon":
        from rufus.scheduler import main as scheduler_main

        scheduler_main()
        return
    if sub == "report":
        from rufus.report_cli import main as report_main

        report_main(rest)
        return
    if sub == "paper":
        from rufus.paper_cli import main as paper_main

        paper_main()
        return
    if sub == "ticker":
        from rufus.ticker_cli import main as ticker_main

        ticker_main(rest)
        return
    from rufus.shell import shell_main

    shell_main()


if __name__ == "__main__":
    main()
"""Entry point: python -m inpaint_lab {run,serve,compare} [options]."""

import importlib
import sys

COMMANDS = {
    "run": ("inpaint_lab.runner", "Generate one collage"),
    "serve": ("inpaint_lab.server", "Start the review and batch web app"),
    "compare": ("inpaint_lab.compare", "Correlate automatic scores with human reviews"),
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        lines = [f"  {name:<8} {description}" for name, (_, description) in COMMANDS.items()]
        sys.exit("usage: python -m inpaint_lab COMMAND [options]\n\ncommands:\n" + "\n".join(lines)
                 + "\n\nUse 'python -m inpaint_lab COMMAND --help' for command options.")
    command = sys.argv.pop(1)
    sys.argv[0] = f"python -m inpaint_lab {command}"
    importlib.import_module(COMMANDS[command][0]).main()


if __name__ == "__main__":
    main()

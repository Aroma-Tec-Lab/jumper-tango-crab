"""Install this project's task into a checkout of KingKongRobotics/jumper.

    python jumper-tasks/install.py <path/to/jumper>

Copies jumper.dance_tango into <jumper>/tasks/jumper/, adds the one import line it needs in
tasks/__init__.py (a task that is not imported there silently does not exist), and lists it in
NOT_LOCOMOTION in tests/test_task_parity.py (an imitation task, not a gait control). Idempotent.
"""
import shutil
import sys
from pathlib import Path

TASKS = ("dance_tango",)
HERE = Path(__file__).resolve().parent


def main(jumper: Path) -> None:
    for t in TASKS:
        dst = jumper / "tasks" / "jumper" / t
        shutil.copytree(HERE / t, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    init = jumper / "tasks" / "__init__.py"
    s = init.read_text()
    anchor = "from .jumper import dance  # noqa: F401,E402\n"
    assert anchor in s, f"{init}: anchor line not found"
    for t in TASKS:
        line = f"from .jumper import {t}  # noqa: F401,E402\n"
        if line not in s:
            s = s.replace(anchor, anchor + line, 1)
    init.write_text(s)
    parity = jumper / "tests" / "test_task_parity.py"
    p = parity.read_text()
    anchor = "NOT_LOCOMOTION = {\n"
    assert anchor in p, f"{parity}: NOT_LOCOMOTION not found"
    for t in TASKS:
        entry = f'    "jumper.{t}",\n'
        if entry not in p:
            p = p.replace(anchor, anchor + entry, 1)
    parity.write_text(p)
    print("installed:", ", ".join(f"jumper.{t}" for t in TASKS))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())

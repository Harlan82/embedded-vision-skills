#!/usr/bin/env python3
"""Install embedded-vision-skills into a local skills directory.

The repo groups skills by domain (skills/model-convert/...), but most skill
loaders require the directory name to equal the frontmatter `name`. This
installer flattens the layout on copy.

Usage:
    python install.py --list
    python install.py
    python install.py --target ~/.claude/skills
    python install.py --only pt-to-onnx-export,yolo-train-tuning
    python install.py --dry-run
"""
import argparse
import shutil
import sys
from pathlib import Path

REPO_SKILLS = Path(__file__).resolve().parent / "skills"
DEFAULT_TARGET = Path.home() / ".workbuddy" / "skills"
IGNORE = ["__pycache__", "*.pyc", "*.pyo"]


def discover():
    """Return [(skill_name, area, source_dir)] for every SKILL.md in the repo."""
    found = []
    for md in sorted(REPO_SKILLS.rglob("SKILL.md")):
        src = md.parent
        area = src.relative_to(REPO_SKILLS).parts[0] if src != REPO_SKILLS else "-"
        found.append((src.name, area, src))
    return found


def main():
    ap = argparse.ArgumentParser(
        description="Install embedded-vision-skills (flattened) into a skills directory."
    )
    ap.add_argument("--target", help="destination skills dir (default: ~/.workbuddy/skills)")
    ap.add_argument("--only", help="comma-separated skill names to install")
    ap.add_argument("--force", action="store_true", help="overwrite existing skill dirs")
    ap.add_argument("--dry-run", action="store_true", help="show what would happen")
    ap.add_argument("--list", action="store_true", help="list available skills and exit")
    args = ap.parse_args()

    skills = discover()
    if not skills:
        sys.exit("error: no skills found under %s" % REPO_SKILLS)

    if args.list:
        width = max(len(a) for _, a, _ in skills)
        for name, area, _ in skills:
            print("  %-*s  %s" % (width, area, name))
        print("\n%d skills" % len(skills))
        return

    only = {s.strip() for s in args.only.split(",") if s.strip()} if args.only else None
    if only:
        unknown = only - {n for n, _, _ in skills}
        if unknown:
            sys.exit("error: unknown skill(s): %s" % ", ".join(sorted(unknown)))

    target = Path(args.target).expanduser().resolve() if args.target else DEFAULT_TARGET
    if not args.dry_run:
        target.mkdir(parents=True, exist_ok=True)

    verb = "plan" if args.dry_run else "copy"
    installed = skipped = 0

    for name, area, src in skills:
        if only and name not in only:
            continue
        dst = target / name
        if dst.exists() and not args.force:
            print("  skip  %-36s (exists, use --force to overwrite)" % name)
            skipped += 1
            continue
        print("  %s  %-36s <- %s" % (verb, name, area))
        if not args.dry_run:
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*IGNORE))
        installed += 1

    print("\ntarget: %s" % target)
    print("%s: %d   skipped: %d" % ("planned" if args.dry_run else "installed", installed, skipped))


if __name__ == "__main__":
    main()

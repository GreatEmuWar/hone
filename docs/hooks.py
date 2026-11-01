"""MkDocs hook that builds pages from the files already in the repository.

The README, CHANGELOG, benchmark results and examples stay the single source of their text;
docs pages pull them in with a directive on a line of its own:

    <!-- include README.md#constraints level=2 -->

- `FILE` includes the whole file; `FILE#slug` only the section whose heading has that slug,
  heading included, up to the next heading at its level or above.
- `level=N` moves the included headings so the first one is at level N.
- `notitle` drops the first heading; `intro` stops at the first heading after the start.
- A `.py` file becomes its module docstring as text, then the rest of its source as code.

Links in included text are written relative to the file they came from. They are rewritten to
the site's page for that file when it has one, and to the file on GitHub otherwise.
"""
import ast
import posixpath
import re
from pathlib import Path

from mkdocs.exceptions import PluginError

ROOT = Path(__file__).resolve().parent.parent
GITHUB = "https://github.com/GreatEmuWar/hone/blob/main/"

# Repository files that have a page of their own on the site.
PAGES = {
    "CHANGELOG.md": "changelog.md",
    "benchmarks/README.md": "benchmarks/index.md",
    "benchmarks/results.md": "benchmarks/results.md",
    "benchmarks/noisy-results.md": "benchmarks/noisy-results.md",
    "benchmarks/ml-results.md": "benchmarks/ml-results.md",
    "benchmarks/constrained-results.md": "benchmarks/constrained-results.md",
    "benchmarks/conditional-results.md": "benchmarks/conditional-results.md",
    "benchmarks/multi-results.md": "benchmarks/multi-results.md",
    "examples/cantilever_beam.py": "examples/cantilever_beam.md",
    "examples/heat_sink.py": "examples/heat_sink.md",
    "examples/choose_a_model.py": "examples/choose_a_model.md",
    "examples/maintenance_schedule.py": "examples/maintenance_schedule.md",
    "examples/pressure_vessel.py": "examples/pressure_vessel.md",
    "examples/tune_iris.py": "examples/tune_iris.md",
}

DIRECTIVE = re.compile(r"^<!-- include (?P<path>[^\s#]+)(?:#(?P<slug>\S+))?(?P<options>(?: \S+)*) -->$", re.M)
FENCE = re.compile(r"^\s*(```|~~~)")
HEADING = re.compile(r"^(#{1,6}) (.*)$")
LINK = re.compile(r"\]\((?P<target>[^)\s]+)\)")


def on_page_markdown(markdown, page, **kwargs):
    def include(match):
        options = dict(o.partition("=")[::2] for o in match["options"].split())
        text = _read(match["path"], match["slug"], options)
        return _rewrite_links(text, match["path"], page.file.src_uri)

    return DIRECTIVE.sub(include, markdown)


def _read(path, slug, options):
    file = ROOT / path
    if not file.is_file():
        raise PluginError(f"include: {path} doesn't exist")
    if file.suffix == ".py":
        return _python(file.read_text())
    lines = file.read_text().splitlines()
    headings = _headings(lines)
    if slug:
        starts = [(i, level) for i, level, title in headings if _slugify(title) == slug]
        if not starts:
            raise PluginError(f"include: {path} has no section #{slug}")
        start, top = starts[0]
        stop = "intro" in options
        end = next((i for i, level, _ in headings if i > start and (stop or level <= top)), len(lines))
        lines = lines[start:end]
    if "notitle" in options:
        first = next((i for i, _, _ in _headings(lines)), None)
        if first is not None:
            del lines[first]
    if "level" in options:
        found = _headings(lines)
        if found:
            shift = int(options["level"]) - found[0][1]
            for i, level, title in found:
                lines[i] = "#" * min(max(level + shift, 1), 6) + " " + title
    return "\n".join(lines).strip() + "\n"


def _headings(lines):
    """(line index, level, title) of each heading outside code blocks."""
    found, fenced = [], False
    for i, line in enumerate(lines):
        if FENCE.match(line):
            fenced = not fenced
        elif not fenced and (match := HEADING.match(line)):
            found.append((i, len(match[1]), match[2].strip()))
    return found


def _slugify(title):
    """GitHub's anchor for a heading, which the README's own links use."""
    title = re.sub(r"[^\w\- ]", "", title.lower())
    return title.replace(" ", "-")


def _python(source):
    module = ast.parse(source)
    docstring = ast.get_docstring(module)
    code = source
    if docstring and module.body:
        code = "\n".join(source.splitlines()[module.body[0].end_lineno:])
    parts = [docstring] if docstring else []
    parts.append("```python\n" + code.strip("\n") + "\n```")
    return "\n\n".join(parts) + "\n"


def _rewrite_links(text, source, page):
    base = posixpath.dirname(source)

    def rewrite(match):
        target = match["target"]
        if re.match(r"[a-z]+:|#|/", target):
            return match[0]
        path, _, anchor = target.partition("#")
        path = posixpath.normpath(posixpath.join(base, path))
        if path in PAGES:
            url = posixpath.relpath(PAGES[path], posixpath.dirname(page) or ".")
        else:
            url = GITHUB + path
        return f"]({url}{'#' + anchor if anchor else ''})"

    out, fenced = [], False
    for line in text.splitlines(keepends=True):
        if FENCE.match(line):
            fenced = not fenced
        out.append(line if fenced else LINK.sub(rewrite, line))
    return "".join(out)

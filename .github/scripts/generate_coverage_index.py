#!/usr/bin/env python3
"""
Generates index.html for the gh-pages coverage history site, plus a
per-run report.html detail page (with sidebar nav + iframe) for each run.

Usage:
    python generate_coverage_index.py <gh-pages-dir> [--tag-map sha1=tag1 sha2=tag2 ...]

The script scans <gh-pages-dir>/reports/ for subdirectories named
YYYY-MM-DD_HH-MM-SS_<sha>, keeps the 5 most recent (unpinned), and writes:
  <gh-pages-dir>/index.html
  <gh-pages-dir>/reports/<run>/report.html   (one per run)

Reports whose SHA appears in --tag-map are never pruned and are shown with
a "Release vX.Y.Z" badge on the index page.

Each report directory is expected to contain:
    branch                             — plain text file with the branch name
    cpp/coverage.html                  — gcovr HTML report
    cpp/coverage-summary.json          — gcovr JSON summary
    python/index.html                  — pytest-cov HTML report
    cuda/index.html                    — genhtml CUDA coverage report (optional)
    cuda/coverage-summary.json         — gpucov JSON summary (optional)
    valgrind/vg_pytest/index.html      — ValgrindCI memcheck report (pytest)
    valgrind/vg_pytest/summary.txt     — valgrind-ci summary text
    valgrind/vg_pytest/raw.log         — raw valgrind text output
    valgrind/vg_pytest/raw.xml         — raw valgrind XML
    valgrind/vg_gtest/  (same layout)
    helgrind/hg_pytest/ (same layout, without raw.xml in new reports)
    helgrind/hg_gtest/  (same layout, without raw.xml in new reports)

Template files are resolved relative to this script's location:
    .github/ci/coverage/index.template.html
    .github/ci/coverage/report.template.html
    .github/ci/coverage/style.css
"""

import json
import os
import re
import shutil
import sys

MAX_REPORTS = 5
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(SCRIPT_DIR, "..", "ci", "coverage")


def load_template(name: str) -> str:
    path = os.path.join(TEMPLATES_DIR, name)
    with open(path) as f:
        return f.read()


def _dir_sha(name: str) -> str | None:
    """Extract the short SHA from a report directory name like YYYY-MM-DD_HH-MM-SS_<sha>."""
    m = re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_([0-9a-f]+)$", name)
    return m.group(1) if m else None


def prune_old_reports(reports_root: str, tag_map: dict[str, str] = {}) -> None:
    if not os.path.isdir(reports_root):
        return
    dirs = sorted(
        [
            d
            for d in os.listdir(reports_root)
            if os.path.isdir(os.path.join(reports_root, d))
        ],
        reverse=True,
    )
    unpinned = [d for d in dirs if _dir_sha(d) not in tag_map]
    for old in unpinned[MAX_REPORTS:]:
        old_path = os.path.join(reports_root, old)
        print(f"Removing old report: {old_path}")
        shutil.rmtree(old_path)


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read()
    except OSError:
        return ""


def parse_cpp_coverage(report_dir: str) -> str | None:
    """Return line coverage % string from gcovr JSON summary, e.g. '64.6%'."""
    path = os.path.join(report_dir, "cpp", "coverage-summary.json")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        pct = data.get("line_percent") or data.get("lines", {}).get("percent")
        if pct is not None:
            return f"{pct:.1f}%"
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        pass
    return None


def parse_cuda_coverage(report_dir: str) -> str | None:
    """Return line coverage % string from CUDA coverage-summary.json, e.g. '89.1%'."""
    path = os.path.join(report_dir, "cuda", "coverage-summary.json")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        pct = data.get("line_percent")
        if pct is not None:
            return f"{pct:.1f}%"
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        pass
    return None


def parse_python_coverage(report_dir: str) -> str | None:
    """Return total coverage % string from coverage.py HTML, e.g. '69%'."""
    html = _read(os.path.join(report_dir, "python", "index.html"))
    m = re.search(r'class=["\']pc_cov["\']>\s*([\d.]+%)\s*<', html)
    return m.group(1) if m else None


def parse_valgrind_error_count(report_dir: str, subpath: str) -> int | None:
    """Return integer error count from a ValgrindCI HTML report, or None if unavailable."""
    html = _read(os.path.join(report_dir, subpath, "index.html"))
    m = re.search(r"<b>(\d+)</b>\s*errors?", html)
    return int(m.group(1)) if m else None


def _has_memory_reports(report_dir: str) -> bool:
    for subpath in (
        "valgrind/vg_pytest",
        "valgrind/vg_gtest",
        "helgrind/hg_pytest",
        "helgrind/hg_gtest",
    ):
        if os.path.isdir(os.path.join(report_dir, subpath)):
            return True
    return False


def get_entries(reports_root: str, tag_map: dict[str, str] = {}) -> list[dict]:
    entries = []
    if not os.path.isdir(reports_root):
        return entries

    for name in sorted(os.listdir(reports_root), reverse=True):
        report_dir = os.path.join(reports_root, name)
        if not os.path.isdir(report_dir):
            continue
        m = re.match(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_([0-9a-f]+)", name)
        if m:
            date_str = m.group(1)
            time_str = m.group(2).replace("-", ":")
            sha = m.group(3)
            utc_iso = f"{date_str}T{time_str}Z"
            label = f"{date_str} {time_str} UTC \u2014 {sha}"
        else:
            utc_iso = ""
            label = name
            sha = ""

        branch = _read(os.path.join(report_dir, "branch")).strip() or None
        release_tag = tag_map.get(sha) if sha else None

        vg_pytest_errs = parse_valgrind_error_count(report_dir, "valgrind/vg_pytest")
        vg_gtest_errs = parse_valgrind_error_count(report_dir, "valgrind/vg_gtest")
        hg_pytest_errs = parse_valgrind_error_count(report_dir, "helgrind/hg_pytest")
        hg_gtest_errs = parse_valgrind_error_count(report_dir, "helgrind/hg_gtest")

        entries.append(
            {
                "name": name,
                "label": label,
                "utc_iso": utc_iso,
                "sha": sha,
                "branch": branch,
                "release_tag": release_tag,
                "has_memory": _has_memory_reports(report_dir),
                "cpp_coverage": parse_cpp_coverage(report_dir),
                "python_coverage": parse_python_coverage(report_dir),
                "cuda_coverage": parse_cuda_coverage(report_dir),
                "vg_pytest_errors": vg_pytest_errs,
                "vg_gtest_errors": vg_gtest_errs,
                "hg_pytest_errors": hg_pytest_errs,
                "hg_gtest_errors": hg_gtest_errs,
                # Paths relative to gh-pages root (used in index.html)
                "detail_path": f"reports/{name}/report.html",
                "cpp_path": f"reports/{name}/cpp/coverage.html",
                "python_path": f"reports/{name}/python/index.html",
                "cuda_path": f"reports/{name}/cuda/html/index.html",
                # Paths relative to the run dir (used in report.html iframe srcs)
                "cpp_rel": "cpp/coverage.html",
                "cpp_detailed_rel": "cpp/detailed/index.html",
                "python_rel": "python/index.html",
                "cuda_rel": "cuda/html/index.html",
                "vg_pytest_rel_base": "valgrind/vg_pytest",
                "vg_gtest_rel_base": "valgrind/vg_gtest",
                "hg_pytest_rel_base": "helgrind/hg_pytest",
                "hg_gtest_rel_base": "helgrind/hg_gtest",
            }
        )
    return entries


# ── Badge helpers ──────────────────────────────────────────────────────────────


def _cov_badge_index(value: str | None) -> str:
    """Small coverage % badge for the index page cards."""
    if value is None:
        return ""
    return f'<span class="stat stat-ok">{value}</span>'


def _err_badge_index(count: int | None) -> str:
    """Small error count badge for the index page cards."""
    if count is None:
        return ""
    label = f"{count} err{'s' if count != 1 else ''}"
    cls = "stat-error" if count > 0 else "stat-ok"
    return f'<span class="stat {cls}">{label}</span>'


def _cov_badge_sidebar(value: str | None) -> str:
    """Coverage badge for the detail page sidebar."""
    if value is None:
        return ""
    return f'<span class="sidebar-section-badge badge-ok">{value}</span>'


def _err_badge_sidebar(count: int | None) -> str:
    """Error count badge for the detail page sidebar nav items."""
    if count is None:
        return ""
    cls = "badge-err" if count > 0 else "badge-ok"
    label = f"{count} err{'s' if count != 1 else ''}"
    return f'<span class="sidebar-section-badge {cls}">{label}</span>'


def _mem_section_badge(counts: list[int | None]) -> str:
    """Aggregate badge for a whole memory section header."""
    known = [c for c in counts if c is not None]
    if not known:
        return '<span class="sidebar-section-badge badge-none">—</span>'
    total = sum(known)
    cls = "badge-err" if total > 0 else "badge-ok"
    label = f"{total} err{'s' if total != 1 else ''}"
    return f'<span class="sidebar-section-badge {cls}">{label}</span>'


# ── Index page rendering ───────────────────────────────────────────────────────


def render_cards(entries: list[dict]) -> str:
    if not entries:
        return '<div class="empty">No coverage reports found yet.</div>'

    cards = []
    for i, entry in enumerate(entries):
        is_latest = i == 0
        is_release = bool(entry.get("release_tag"))
        wrapper_class = (
            "report-card-wrapper"
            + (" latest" if is_latest else "")
            + (" release" if is_release else "")
        )
        badges = ""
        if is_latest:
            badges += '<span class="badge badge-latest">Latest</span>'
        if is_release:
            badges += f'<span class="badge badge-release">&#x1f3f7; {entry["release_tag"]}</span>'
        branch_tag = (
            f'<div class="report-branch">&#x2387; {entry["branch"]}</div>'
            if entry.get("branch")
            else ""
        )

        # Only valgrind (memcheck) errors on the main page — helgrind counts
        # data-race errors which are a different category shown in the detail view
        vg_counts = [entry["vg_pytest_errors"], entry["vg_gtest_errors"]]
        known = [c for c in vg_counts if c is not None]
        if known:
            total_mem = sum(known)
            mem_cls = "stat-error" if total_mem > 0 else "stat-ok"
            mem_label = f"{total_mem} mem err{'s' if total_mem != 1 else ''}"
            mem_stat = f'<span class="stat {mem_cls}">{mem_label}</span>'
        else:
            mem_stat = ""

        cards.append(f"""
<div class="{wrapper_class}">
  <a class="report-card-link" href="{entry["detail_path"]}">
    <div class="report-card">
      <span class="report-index">#{i + 1}</span>
      <div class="report-info">
        <div class="report-label" data-utc="{entry["utc_iso"]}" data-sha="{entry["sha"]}">{entry["label"]}</div>
        {branch_tag}
      </div>
      {badges}
      <div class="report-stats">
        {_cov_badge_index(entry["cpp_coverage"]) and f'<span class="stat-group"><span class="stat-label">C++</span>{_cov_badge_index(entry["cpp_coverage"])}</span>' or ""}
        {_cov_badge_index(entry["python_coverage"]) and f'<span class="stat-group"><span class="stat-label">Py</span>{_cov_badge_index(entry["python_coverage"])}</span>' or ""}
        {_cov_badge_index(entry["cuda_coverage"]) and f'<span class="stat-group"><span class="stat-label">CUDA</span>{_cov_badge_index(entry["cuda_coverage"])}</span>' or ""}
        {mem_stat}
      </div>
      <div class="card-arrow">&#x2192;</div>
    </div>
  </a>
</div>""")

    return "\n".join(cards)


# (no CHART_CSS constant — chart styles live in style.css)


def load_coverage_history(gh_pages_dir: str) -> list[dict]:
    """Load the persistent coverage history JSON, or return empty list."""
    path = os.path.join(gh_pages_dir, "coverage-history.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return []


def save_coverage_history(gh_pages_dir: str, history: list[dict]) -> None:
    """Write the coverage history JSON, sorted oldest-first by utc_iso."""
    history_sorted = sorted(history, key=lambda d: d.get("utc_iso", ""))
    path = os.path.join(gh_pages_dir, "coverage-history.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history_sorted, f, indent=2)
    print(f"Written coverage history ({len(history_sorted)} entries): {path}")


def reconcile_coverage_history(
    gh_pages_dir: str, entries: list[dict], tag_map: dict[str, str]
) -> None:
    """Upsert all current entries into the persistent history file and refresh
    release_tag for every record against the latest tag_map.  Pruned runs that
    were already written to the file are preserved — only their release_tag is
    updated if a tag now points at them."""
    history = load_coverage_history(gh_pages_dir)
    by_sha: dict[str, dict] = {h["sha"]: h for h in history if h.get("sha")}

    # Upsert entries that are still present as report directories
    for entry in entries:
        cpp = entry.get("cpp_coverage")
        py = entry.get("python_coverage")
        cuda = entry.get("cuda_coverage")
        if cpp is None and py is None and cuda is None:
            continue
        sha = entry.get("sha", "")
        record = {
            "utc_iso": entry["utc_iso"],
            "sha": sha,
            "branch": entry.get("branch"),
            "detail_path": entry.get("detail_path"),
            "cpp": float(cpp.rstrip("%")) if cpp else None,
            "py": float(py.rstrip("%")) if py else None,
            "cuda": float(cuda.rstrip("%")) if cuda else None,
            "release_tag": tag_map.get(sha) if sha else None,
        }
        by_sha[sha] = record

    # Refresh release_tag for pruned runs that are still in history
    for sha, record in by_sha.items():
        record["release_tag"] = tag_map.get(sha) or record.get("release_tag")

    save_coverage_history(gh_pages_dir, list(by_sha.values()))


def render_coverage_chart(gh_pages_dir: str, entries: list[dict]) -> str:
    """Render a canvas coverage history chart, reading from the persistent history file.
    Falls back to deriving points from live entries if no history file exists yet."""
    history = load_coverage_history(gh_pages_dir)

    if history:
        points = history  # already oldest-first after save_coverage_history sort
    else:
        # Fallback: derive from whatever report dirs are still present
        points = []
        for entry in reversed(entries):
            cpp = entry.get("cpp_coverage")
            py = entry.get("python_coverage")
            cuda = entry.get("cuda_coverage")
            if cpp is None and py is None and cuda is None:
                continue
            points.append(
                {
                    "utc_iso": entry["utc_iso"],
                    "sha": entry["sha"],
                    "detail_path": entry.get("detail_path"),
                    "cpp": float(cpp.rstrip("%")) if cpp else None,
                    "py": float(py.rstrip("%")) if py else None,
                    "cuda": float(cuda.rstrip("%")) if cuda else None,
                    "release_tag": entry.get("release_tag"),
                }
            )

    if not points:
        return ""

    return load_template("chart.template.html").replace("{{data}}", json.dumps(points))


def render_index_html(gh_pages_dir: str, entries: list[dict]) -> str:
    template = load_template("index.template.html")
    css = load_template("style.css")
    cards = render_cards(entries)
    chart = render_coverage_chart(gh_pages_dir, entries)
    return (
        template.replace("{{style}}", css)
        .replace("{{chart}}", chart)
        .replace("{{cards}}", cards)
        .replace("{{max_reports}}", str(MAX_REPORTS))
    )


# ── Detail page rendering ──────────────────────────────────────────────────────


def _memory_nav_section(entry: dict, report_dir: str) -> str:
    """Render the Valgrind + Helgrind sidebar sections for report.html."""
    if not entry["has_memory"]:
        return ""

    def _group_items(base: str, errors: int | None) -> str:
        err_badge = _err_badge_sidebar(errors)
        supp_path = os.path.join(report_dir, base, "generated.supp")
        has_supp = os.path.isfile(supp_path) and os.path.getsize(supp_path) > 0
        xml_row = ""
        if os.path.isfile(os.path.join(report_dir, base, "raw.xml")):
            xml_row = f"""
        <div class="sidebar-nav-row">
          <a class="sidebar-nav-item" data-src="{base}/raw.xml" data-mode="xml" href="#">
            <span class="nav-icon">&#x1f5c2;</span> raw.xml
          </a>
          <a class="sidebar-dl-btn" href="{base}/raw.xml" download title="Download">&#x2913;</a>
        </div>"""
        supp_row = ""
        if has_supp:
            supp_row = f"""
        <div class="sidebar-nav-row">
          <a class="sidebar-nav-item" data-src="{base}/generated.supp" data-mode="text" href="#">
            <span class="nav-icon">&#x1f6e1;</span> generated.supp
          </a>
          <a class="sidebar-dl-btn" href="{base}/generated.supp" download title="Download">&#x2913;</a>
        </div>"""
        return f"""
        <div class="sidebar-nav-row">
          <a class="sidebar-nav-item" data-src="{base}/index.html" href="#">
            <span class="nav-icon">&#x1f4ca;</span> HTML report {err_badge}
          </a>
          <a class="sidebar-dl-btn" href="{base}/index.html" download title="Download">&#x2913;</a>
        </div>
        <div class="sidebar-nav-row">
          <a class="sidebar-nav-item" data-src="{base}/summary.txt" href="#">
            <span class="nav-icon">&#x1f4cb;</span> summary.txt
          </a>
          <a class="sidebar-dl-btn" href="{base}/summary.txt" download title="Download">&#x2913;</a>
        </div>
        <div class="sidebar-nav-row">
          <a class="sidebar-nav-item" data-src="{base}/raw.log" data-mode="text" href="#">
            <span class="nav-icon">&#x1f4dd;</span> raw.log
          </a>
          <a class="sidebar-dl-btn" href="{base}/raw.log" download title="Download">&#x2913;</a>
        </div>{xml_row}{supp_row}"""

    vg_counts = [entry["vg_pytest_errors"], entry["vg_gtest_errors"]]
    hg_counts = [entry["hg_pytest_errors"], entry["hg_gtest_errors"]]

    return f"""
    <!-- Valgrind section -->
    <div class="sidebar-section open">
      <div class="sidebar-section-header">
        <span class="sidebar-section-title">Valgrind (memcheck)</span>
        {_mem_section_badge(vg_counts)}
        <span class="sidebar-chevron">&#x25b8;</span>
      </div>
      <div class="sidebar-section-body">
        <div class="sidebar-subsection-label">pytest</div>
        {_group_items(entry["vg_pytest_rel_base"], entry["vg_pytest_errors"])}
        <div class="sidebar-subsection-label">gtest</div>
        {_group_items(entry["vg_gtest_rel_base"], entry["vg_gtest_errors"])}
      </div>
    </div>

    <!-- Helgrind section -->
    <div class="sidebar-section open">
      <div class="sidebar-section-header">
        <span class="sidebar-section-title">Helgrind</span>
        {_mem_section_badge(hg_counts)}
        <span class="sidebar-chevron">&#x25b8;</span>
      </div>
      <div class="sidebar-section-body">
        <div class="sidebar-subsection-label">pytest</div>
        {_group_items(entry["hg_pytest_rel_base"], entry["hg_pytest_errors"])}
        <div class="sidebar-subsection-label">gtest</div>
        {_group_items(entry["hg_gtest_rel_base"], entry["hg_gtest_errors"])}
      </div>
    </div>"""


def _cuda_nav_section(entry: dict) -> str:
    """Render the CUDA coverage sidebar section for report.html."""
    if not entry.get("cuda_coverage"):
        return ""

    cuda_icon = (
        '<svg width="14" height="14" viewBox="0 0 64 64" style="vertical-align:middle">'
        '<path d="M32 4 C12 4 2 32 2 32 C2 32 12 60 32 60 C52 60 62 32 62 32 C62 32 52 4 32 4Z" fill="#76b900"/>'
        '<ellipse cx="32" cy="32" rx="12" ry="16" fill="#fff" opacity="0.9"/>'
        '<ellipse cx="32" cy="32" rx="6" ry="10" fill="#76b900"/>'
        '</svg>'
    )

    return f"""
        <div class="sidebar-nav-row">
          <a class="sidebar-nav-item" data-src="{entry["cuda_rel"]}" href="#">
            <span class="nav-icon">{cuda_icon}</span> CUDA {_cov_badge_sidebar(entry["cuda_coverage"])}
          </a>
        </div>"""


def render_detail_html(entry: dict, gh_pages_dir: str) -> str:
    template = load_template("report.template.html")
    branch_display = f"&#x2387; {entry['branch']}" if entry.get("branch") else ""
    report_dir = os.path.join(gh_pages_dir, "reports", entry["name"])

    return (
        template.replace(
            "{{page_title}}", f"Report \u2014 {entry['sha'] or entry['name']}"
        )
        .replace("{{utc_iso}}", entry["utc_iso"])
        .replace("{{sha}}", entry["sha"])
        .replace("{{label}}", entry["label"])
        .replace("{{branch_display}}", branch_display)
        .replace("{{cpp_path}}", entry["cpp_rel"])
        .replace("{{cpp_detailed_path}}", entry["cpp_detailed_rel"])
        .replace("{{python_path}}", entry["python_rel"])
        .replace("{{cpp_badge}}", _cov_badge_sidebar(entry["cpp_coverage"]))
        .replace("{{python_badge}}", _cov_badge_sidebar(entry["python_coverage"]))
        .replace("{{cuda_nav}}", _cuda_nav_section(entry))
        .replace("{{memory_nav}}", _memory_nav_section(entry, report_dir))
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    if len(sys.argv) < 2:
        print(
            f"Usage: {sys.argv[0]} <gh-pages-dir> [--tag-map sha1=tag1 sha2=tag2 ...]",
            file=sys.stderr,
        )
        sys.exit(1)

    gh_pages_dir = sys.argv[1]
    reports_root = os.path.join(gh_pages_dir, "reports")

    # Parse --tag-map sha=tag pairs (e.g. abc1234=v0.4.0)
    tag_map: dict[str, str] = {}
    args = sys.argv[2:]
    if "--tag-map" in args:
        idx = args.index("--tag-map")
        for item in args[idx + 1 :]:
            if "=" in item:
                sha, tag = item.split("=", 1)
                tag_map[sha] = tag

    if tag_map:
        print("Tagged releases (will not be pruned):")
        for sha, tag in sorted(tag_map.items()):
            print(f"  {sha} -> {tag}")

    prune_old_reports(reports_root, tag_map)
    entries = get_entries(reports_root, tag_map)

    # Upsert surviving entries into the persistent history file and refresh
    # release tags. Pruned entries already in the file are preserved.
    reconcile_coverage_history(gh_pages_dir, entries, tag_map)

    # Write per-run detail pages
    for entry in entries:
        detail_html = render_detail_html(entry, gh_pages_dir)
        out_path = os.path.join(gh_pages_dir, "reports", entry["name"], "report.html")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(detail_html)
        print(f"  Written detail page: {out_path}")

    # Write main index
    index_html = render_index_html(gh_pages_dir, entries)
    out_path = os.path.join(gh_pages_dir, "index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(index_html)

    print(f"Written {out_path} with {len(entries)} report(s).")


if __name__ == "__main__":
    main()

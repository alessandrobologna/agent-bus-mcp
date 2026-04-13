#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import posixpath
import re
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

SECTION_ORDER = ("tutorials", "how-to", "reference", "explanation")
EXCLUDED_DOCS = {PurePosixPath("docs/diataxis-migration-matrix.md")}
SPECIAL_PAGES = {
    PurePosixPath("spec.md"): PurePosixPath("reference/implementation-spec.mdx"),
    PurePosixPath("CHANGELOG.md"): PurePosixPath("reference/changelog.mdx"),
}
ASSET_SOURCE_ROOT = PurePosixPath("docs/images")
ASSET_TARGET_ROOT = PurePosixPath("docs-assets/images")

FENCE_RE = re.compile(r"(?ms)^```.*?^```[ \t]*\n?")
MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"([^\"]*)\")?\)")
MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)\s]+)(?:\s+\"([^\"]*)\")?\)")
HTML_IMAGE_RE = re.compile(r"(?is)<img\b([^>]*?)\/?>")
ATTR_RE = re.compile(r'([A-Za-z_:][-A-Za-z0-9_:.]*)\s*=\s*"([^"]*)"')


@dataclass(frozen=True)
class PageRecord:
    source_rel: PurePosixPath
    target_rel: PurePosixPath
    title: str
    description: str | None


@dataclass(frozen=True)
class SectionInfo:
    title: str
    description: str | None


def normalize_repo_rel(path: PurePosixPath) -> PurePosixPath:
    normalized = posixpath.normpath(path.as_posix())
    if normalized == ".":
        return PurePosixPath(".")
    return PurePosixPath(normalized)


def resolve_repo_rel(base_dir: PurePosixPath, href: str) -> PurePosixPath | None:
    if not href or href.startswith(("#", "http://", "https://", "mailto:", "data:")):
        return None
    if href.startswith("/"):
        return normalize_repo_rel(PurePosixPath(href.lstrip("/")))
    resolved = normalize_repo_rel(base_dir / href)
    if resolved.parts and resolved.parts[0] == "..":
        return None
    return resolved


def replace_outside_code_fences(text: str, replacer) -> str:
    pieces: list[str] = []
    last = 0
    for match in FENCE_RE.finditer(text):
        pieces.append(replacer(text[last : match.start()]))
        pieces.append(match.group(0))
        last = match.end()
    pieces.append(replacer(text[last:]))
    return "".join(pieces)


def strip_leading_h1(text: str) -> str:
    lines = text.splitlines()
    index = 0
    while index < len(lines) and not lines[index].strip():
        index += 1
    if index < len(lines) and lines[index].startswith("# "):
        index += 1
        while index < len(lines) and not lines[index].strip():
            index += 1
        return "\n".join(lines[index:]).lstrip("\n")
    return text


def extract_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return fallback


def extract_description(text: str) -> str | None:
    lines = strip_leading_h1(text).splitlines()
    block: list[str] = []
    in_fence = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            if block:
                break
            continue
        if in_fence:
            continue
        if not stripped:
            if block:
                break
            continue
        if stripped.startswith(("#", ">", "-", "*", "|", "<")):
            if block:
                break
            continue
        block.append(stripped)
    if not block:
        return None
    return " ".join(block)


def strip_leading_description_block(text: str, description: str | None) -> str:
    if not description:
        return text

    lines = text.splitlines()
    block: list[str] = []
    block_start: int | None = None
    block_end: int | None = None
    in_fence = False

    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            if block:
                block_end = index
                break
            continue
        if in_fence:
            continue
        if not stripped:
            if block:
                block_end = index
                break
            continue
        if stripped.startswith(("#", ">", "-", "*", "|", "<")):
            if block:
                block_end = index
            else:
                return text
            break
        if block_start is None:
            block_start = index
        block.append(stripped)

    if not block or block_start is None:
        return text
    if " ".join(block) != description:
        return text
    if block_end is None:
        block_end = len(lines)
    return "\n".join(lines[block_end:]).lstrip("\n")


def build_frontmatter(title: str, description: str | None) -> str:
    lines = ["---", f"title: {json.dumps(title)}"]
    if description:
        lines.append(f"description: {json.dumps(description)}")
    lines.extend(("---", ""))
    return "\n".join(lines)


def asset_site_path_for(repo_rel: PurePosixPath) -> PurePosixPath | None:
    try:
        suffix = repo_rel.relative_to(ASSET_SOURCE_ROOT)
    except ValueError:
        return None
    return ASSET_TARGET_ROOT / suffix


def site_route_path_for_generated_page(target_rel: PurePosixPath) -> str:
    route = route_path_for_generated_page(target_rel)
    return "docs" if not route else f"docs/{route}"


def relative_asset_href(target_rel: PurePosixPath, repo_rel: PurePosixPath) -> str | None:
    asset_site_path = asset_site_path_for(repo_rel)
    if asset_site_path is None:
        return None
    current_site_route = site_route_path_for_generated_page(target_rel)
    return posixpath.relpath(asset_site_path.as_posix(), start=current_site_route or ".")


def rewrite_html_images(text: str, source_rel: PurePosixPath, target_rel: PurePosixPath) -> str:
    def replace(match: re.Match[str]) -> str:
        attrs = ATTR_RE.findall(match.group(1))
        rewritten: list[tuple[str, str]] = []
        src_found = False
        for key, value in attrs:
            if key == "src":
                resolved = resolve_repo_rel(source_rel.parent, value)
                if resolved:
                    relative = relative_asset_href(target_rel, resolved)
                    if relative:
                        value = relative
                src_found = True
            rewritten.append((key, value))
        if not src_found:
            return match.group(0)
        rendered = " ".join(f'{key}="{value}"' for key, value in rewritten)
        return f"<img {rendered} />"

    return HTML_IMAGE_RE.sub(replace, text)


def rewrite_markdown_images(text: str, source_rel: PurePosixPath, target_rel: PurePosixPath) -> str:
    def replace(match: re.Match[str]) -> str:
        alt, href, title = match.group(1), match.group(2), match.group(3)
        resolved = resolve_repo_rel(source_rel.parent, href)
        if not resolved:
            return match.group(0)
        relative = relative_asset_href(target_rel, resolved)
        if not relative:
            return match.group(0)
        title_suffix = f' "{title}"' if title else ""
        return f"![{alt}]({relative}{title_suffix})"

    return MARKDOWN_IMAGE_RE.sub(replace, text)


def rewrite_markdown_links(
    text: str,
    source_rel: PurePosixPath,
    target_rel: PurePosixPath,
    source_map: dict[PurePosixPath, PurePosixPath],
) -> str:
    current_route = route_path_for_generated_page(target_rel)

    def replace(match: re.Match[str]) -> str:
        label, href, title = match.group(1), match.group(2), match.group(3)
        if href.startswith(("#", "http://", "https://", "mailto:", "data:")):
            return match.group(0)

        raw_path, hash_sep, fragment = href.partition("#")
        resolved = resolve_repo_rel(source_rel.parent, raw_path)
        if not resolved:
            return match.group(0)
        mapped = source_map.get(resolved)
        if mapped is None:
            return match.group(0)
        target_route = route_path_for_generated_page(mapped)
        relative = posixpath.relpath(target_route or ".", start=current_route or ".")
        rewritten = "./" if relative == "." else relative
        if not rewritten.endswith("/"):
            rewritten = f"{rewritten}/"
        if hash_sep:
            rewritten = f"{rewritten}#{fragment}"
        title_suffix = f' "{title}"' if title else ""
        return f"[{label}]({rewritten}{title_suffix})"

    return MARKDOWN_LINK_RE.sub(replace, text)


def rewrite_content(
    text: str,
    source_rel: PurePosixPath,
    target_rel: PurePosixPath,
    source_map: dict[PurePosixPath, PurePosixPath],
) -> str:
    def replacer(segment: str) -> str:
        segment = rewrite_html_images(segment, source_rel, target_rel)
        segment = rewrite_markdown_images(segment, source_rel, target_rel)
        return rewrite_markdown_links(segment, source_rel, target_rel, source_map)

    return replace_outside_code_fences(text, replacer)


def docs_target_for(source_rel: PurePosixPath) -> PurePosixPath:
    relative = source_rel.relative_to("docs")
    if relative.name == "README.md":
        stem = relative.parent / "index.mdx"
    else:
        stem = relative.with_suffix(".mdx")
    return normalize_repo_rel(stem)


def is_section_readme(source_rel: PurePosixPath) -> bool:
    return (
        len(source_rel.parts) == 3
        and source_rel.parts[:1] == ("docs",)
        and source_rel.parts[1] in SECTION_ORDER
        and source_rel.name == "README.md"
    )


def route_path_for_generated_page(target_rel: PurePosixPath) -> str:
    if target_rel == PurePosixPath("index.mdx"):
        return ""
    path = target_rel.with_suffix("").as_posix()
    if path.endswith("/index"):
        return path[: -len("/index")]
    return path


def build_source_map(repo_root: Path) -> dict[PurePosixPath, PurePosixPath]:
    mapping: dict[PurePosixPath, PurePosixPath] = {}
    docs_root = repo_root / "docs"
    for path in sorted(docs_root.rglob("*.md")):
        source_rel = PurePosixPath(path.relative_to(repo_root).as_posix())
        if source_rel in EXCLUDED_DOCS:
            continue
        if is_section_readme(source_rel):
            continue
        mapping[source_rel] = docs_target_for(source_rel)
    mapping.update(SPECIAL_PAGES)
    return mapping


def build_section_info(repo_root: Path) -> dict[str, SectionInfo]:
    info: dict[str, SectionInfo] = {}
    for section in SECTION_ORDER:
        readme_path = repo_root / "docs" / section / "README.md"
        if readme_path.exists():
            text = readme_path.read_text(encoding="utf-8")
            title = extract_title(text, section.replace("-", " ").title())
            description = extract_description(text)
        else:
            title = section.replace("-", " ").title()
            description = None
        info[section] = SectionInfo(title=title, description=description)
    return info


def write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def copy_assets(repo_root: Path, public_root: Path) -> None:
    source_dir = repo_root / ASSET_SOURCE_ROOT
    target_dir = public_root / ASSET_TARGET_ROOT
    if target_dir.exists():
        shutil.rmtree(target_dir, ignore_errors=True)
    if not source_dir.exists():
        return
    for source in sorted(source_dir.rglob("*")):
        if source.is_dir():
            continue
        relative = source.relative_to(source_dir)
        destination = target_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def write_page(
    repo_root: Path,
    content_root: Path,
    record: PageRecord,
    source_map: dict[PurePosixPath, PurePosixPath],
) -> None:
    source_path = repo_root / record.source_rel
    raw = source_path.read_text(encoding="utf-8")
    body = strip_leading_h1(raw)
    body = strip_leading_description_block(body, record.description)
    rewritten = rewrite_content(body, record.source_rel, record.target_rel, source_map).rstrip()
    destination = content_root / record.target_rel
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        build_frontmatter(record.title, record.description) + rewritten + "\n",
        encoding="utf-8",
    )


def build_records(
    repo_root: Path, source_map: dict[PurePosixPath, PurePosixPath]
) -> list[PageRecord]:
    records: list[PageRecord] = []
    for source_rel, target_rel in sorted(source_map.items(), key=lambda item: item[1].as_posix()):
        source_path = repo_root / source_rel
        fallback_title = target_rel.stem.replace("-", " ").title()
        text = source_path.read_text(encoding="utf-8")
        records.append(
            PageRecord(
                source_rel=source_rel,
                target_rel=target_rel,
                title=extract_title(text, fallback_title),
                description=extract_description(text),
            )
        )
    return records


def group_records_by_section(records: list[PageRecord]) -> dict[str, list[PageRecord]]:
    by_section: dict[str, list[PageRecord]] = {section: [] for section in SECTION_ORDER}
    for record in records:
        parts = record.target_rel.parts
        if not parts:
            continue
        section = parts[0]
        if section in by_section:
            by_section[section].append(record)
    return by_section


def ordered_section_pages(section: str, records: list[PageRecord]) -> list[PageRecord]:
    ordered = sorted(records, key=lambda record: record.target_rel.stem)
    if section == "reference":
        priority = {
            "runtime-reference": 0,
            "implementation-spec": 1,
            "changelog": 2,
        }
        ordered.sort(
            key=lambda record: (priority.get(record.target_rel.stem, 99), record.target_rel.stem)
        )
    return ordered


def build_link_map(
    source_map: dict[PurePosixPath, PurePosixPath],
    records: list[PageRecord],
) -> dict[PurePosixPath, PurePosixPath]:
    link_map = dict(source_map)
    by_section = group_records_by_section(records)
    for section in SECTION_ORDER:
        ordered = ordered_section_pages(section, by_section[section])
        if not ordered:
            continue
        link_map[PurePosixPath("docs") / section / "README.md"] = ordered[0].target_rel
    return link_map


def write_meta_files(
    content_root: Path,
    records: list[PageRecord],
    section_info: dict[str, SectionInfo],
) -> None:
    by_section = group_records_by_section(records)
    root_description = None
    for record in records:
        if record.target_rel == PurePosixPath("index.mdx"):
            root_description = record.description

    root_meta: dict[str, object] = {"title": "Documentation", "pages": ["index", *SECTION_ORDER]}
    if root_description:
        root_meta["description"] = root_description
    write_json(content_root / "meta.json", root_meta)

    for section in SECTION_ORDER:
        ordered_pages = ordered_section_pages(section, by_section[section])
        data: dict[str, object] = {
            "title": section_info[section].title,
            "pages": [record.target_rel.stem for record in ordered_pages],
        }
        if section_info[section].description:
            data["description"] = section_info[section].description
        write_json(content_root / section / "meta.json", data)


def generate_site_content(repo_root: Path, content_root: Path, public_root: Path) -> None:
    source_map = build_source_map(repo_root)
    records = build_records(repo_root, source_map)
    section_info = build_section_info(repo_root)
    link_map = build_link_map(source_map, records)

    if content_root.exists():
        shutil.rmtree(content_root)
    content_root.mkdir(parents=True, exist_ok=True)

    for record in records:
        write_page(repo_root, content_root, record, link_map)
    write_meta_files(content_root, records, section_info)
    copy_assets(repo_root, public_root)


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent
    parser = argparse.ArgumentParser(description="Generate Fumadocs content from repository docs.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--content-root", type=Path, default=repo_root / "site" / "content" / "docs"
    )
    parser.add_argument("--public-root", type=Path, default=repo_root / "site" / "public")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generate_site_content(
        repo_root=args.repo_root.resolve(),
        content_root=args.content_root.resolve(),
        public_root=args.public_root.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

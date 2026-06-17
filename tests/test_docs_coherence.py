from pathlib import Path


ACTIVE_DOCS = [
    Path("docs"),
    Path("doc/CALYPSO_RUNBOOK.md"),
    Path("doc/ENVIRONMENT_SETUP.md"),
    Path("doc/HOME_REPO_LAYOUT.md"),
]

STALE_RUNNER_TOKENS = [
    "run_exp5_workflow.py",
    "run_exp5_workflow_grace.sh",
    "submit_exp5_workflow_grace.sh",
    "submit_exp5_workflow_globc.sh",
]

EGU_RELEASE_MARKDOWN = [
    Path("docs/egu26_short_course/SESSION_MATERIALS.md"),
    Path("docs/egu26_short_course/SESSION_SUMMARY.md"),
    Path("docs/egu26_short_course/ENVIRONMENT_SETUP.md"),
    Path("docs/egu26_short_course/HELPER_SCRIPTS.md"),
    Path("docs/egu26_short_course/LOCAL_WORKFLOW_RUNBOOK.md"),
    Path("docs/egu26_short_course/WORKFLOW_PHASES.md"),
]


def iter_active_doc_texts():
    for root in ACTIVE_DOCS:
        if root.is_dir():
            for path in root.rglob("*"):
                if path.suffix not in {".md", ".rst", ".ipynb"}:
                    continue
                yield path, path.read_text(encoding="utf-8")
        else:
            yield root, root.read_text(encoding="utf-8")


def test_active_docs_do_not_reference_stale_runner_names():
    stale_hits = []
    for path, text in iter_active_doc_texts():
        for token in STALE_RUNNER_TOKENS:
            if token in text:
                stale_hits.append(f"{path}: {token}")
    assert not stale_hits, "\n".join(stale_hits)


def test_egu_markdown_pages_include_release_compatibility_note():
    required_note = "Release compatibility: this EGU26 short-course material is maintained against `idownscale` release `v1.4.0`."
    for path in EGU_RELEASE_MARKDOWN:
        text = path.read_text(encoding="utf-8")
        assert required_note in text, str(path)
        assert "git checkout v1.4.0" in text or path.name not in {"SESSION_MATERIALS.md", "ENVIRONMENT_SETUP.md"}


def test_egu_notebook_includes_release_compatibility_and_checkout_guidance():
    text = Path("docs/egu26_short_course/egu26_short_course_notebook.ipynb").read_text(encoding="utf-8")
    assert "Release compatibility: this notebook is maintained against `idownscale` `v1.4.0`." in text
    assert "git checkout v1.4.0" in text

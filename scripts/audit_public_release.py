"""Audit the Git index without printing credentials or changing files.

This is a publication guard, not a legal opinion or a complete secret detector.
It scans the bytes that will be committed, rather than ignored local datasets.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import PurePosixPath

FORBIDDEN_DIRECTORIES = {
    "data", "results", "tmp", "manuscript", "dist", "build", "node_modules",
    ".venv", ".venv-sci", ".qiskit", "__pycache__",
}
FORBIDDEN_EXTENSIONS = {
    ".nc", ".h5", ".hdf5", ".pdf", ".docx", ".zip", ".whl",
    ".pkl", ".pickle", ".joblib", ".pem", ".key", ".token",
}
FORBIDDEN_FILES = {
    "SCI_HANDOFF_KO.md", "risk.csv", "qiskit-ibm.json", "ibm-quantum-account.json",
}
SECRET_PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "github_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "openai_token": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{30,}\b"),
    "possible_ibm_token": re.compile(r"[\"'][A-Za-z0-9_-]{44}[\"']"),
    "literal_credential": re.compile(
        r"(?i)[\"']?(?:api[_-]?key|access[_-]?token|token|password|secret)[\"']?"
        r"\s*[:=]\s*[\"']([^\"'\r\n]{8,})[\"']"
    ),
}


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], stderr=subprocess.PIPE)


def main() -> None:
    manifest = json.loads(git("show", ":PUBLICATION_MANIFEST.json"))
    allowed = set(manifest["reviewed_additions"])
    base = manifest["base_commit"]
    paths = [p for p in git("ls-files", "-z").decode("utf-8").split("\0") if p]
    added = {
        p for p in git("diff", "--cached", "--name-only", "--diff-filter=A", "-z", base)
        .decode("utf-8").split("\0") if p
    }
    findings = []
    for path in sorted(added - allowed):
        findings.append({"path": path, "rule": "unreviewed_addition"})
    for path in paths:
        pure = PurePosixPath(path)
        if (
            set(pure.parts) & FORBIDDEN_DIRECTORIES
            or pure.suffix.lower() in FORBIDDEN_EXTENSIONS
            or pure.name in FORBIDDEN_FILES
            or pure.name == ".env"
            or pure.name.startswith(".env.")
        ):
            findings.append({"path": path, "rule": "excluded_path"})
        payload = git("show", f":{path}")
        if b"\0" in payload:
            findings.append({"path": path, "rule": "binary_payload_requires_review"})
            continue
        text = payload.decode("utf-8", errors="replace")
        for name, pattern in SECRET_PATTERNS.items():
            for match in pattern.finditer(text):
                if name == "possible_ibm_token" and re.fullmatch(
                    r"[a-z]+(?:_[a-z]+){2,}", match.group(0)[1:-1]
                ):
                    # Reviewed scientific column names can also have 44 characters.
                    continue
                if name == "literal_credential":
                    value = match.group(1).lower()
                    if any(word in value for word in ("example", "placeholder", "your_", "your-", "<", ">")):
                        continue
                findings.append({
                    "path": path, "line": text.count("\n", 0, match.start()) + 1,
                    "rule": name,
                })
    print(json.dumps({
        "status": "fail" if findings else "pass",
        "tracked_files_scanned": len(paths),
        "reviewed_additions": len(added),
        "findings": findings,
        "limitation": "Pattern/path checks do not prove rights ownership or absence of every secret",
    }, ensure_ascii=True, indent=2))
    if findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

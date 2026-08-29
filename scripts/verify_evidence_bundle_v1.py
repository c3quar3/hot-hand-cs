# Read-only integrity verifier for the final manuscript evidence bundle.

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "reproducibility" / "final_evidence_bundle_v1.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_package(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    package_root = root / spec["root"]
    errors: list[str] = []
    manifest_path = package_root / spec["manifest"]
    file_manifest_path = package_root / spec["file_manifest"]
    if not manifest_path.is_file():
        return {"name": spec["name"], "status": "fail", "errors": [f"missing_manifest:{manifest_path}"]}
    if not file_manifest_path.is_file():
        return {"name": spec["name"], "status": "fail", "errors": [f"missing_file_manifest:{file_manifest_path}"]}

    manifest, listed = read_json(manifest_path), read_json(file_manifest_path)
    file_manifest_hash = digest(file_manifest_path)
    if manifest.get("status") != spec["expected_status"]:
        errors.append(f"unexpected_manifest_status:{manifest.get('status')}")
    if manifest.get("file_sha256_sha256") != file_manifest_hash:
        errors.append("manifest_file_sha256_chain_mismatch")
    if file_manifest_hash != spec["expected_file_manifest_sha256"]:
        errors.append("bundle_file_manifest_hash_mismatch")

    for relative, expected in spec["required_files"].items():
        path = package_root / relative
        if not path.is_file():
            errors.append(f"missing_required_file:{relative}")
        elif digest(path) != expected:
            errors.append(f"required_file_hash_mismatch:{relative}")
        elif listed.get(relative) != expected:
            errors.append(f"file_manifest_entry_mismatch:{relative}")

    for relative, expected in listed.items():
        path = package_root / relative
        if not path.is_file():
            errors.append(f"missing_listed_file:{relative}")
        elif digest(path) != expected:
            errors.append(f"listed_file_hash_mismatch:{relative}")

    return {
        "name": spec["name"],
        "status": "pass" if not errors else "fail",
        "checked_listed_files": len(listed),
        "checked_required_files": len(spec["required_files"]),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    bundle_path = args.bundle.resolve()
    root = args.root.resolve()
    if not bundle_path.is_file():
        print(json.dumps({"status": "fail", "errors": [f"missing_bundle:{bundle_path}"]}, ensure_ascii=True))
        return 1
    bundle = read_json(bundle_path)
    results = [verify_package(root, spec) for spec in bundle["packages"]]
    output = {
        "stage": bundle["stage"],
        "scope": bundle["scope"],
        "status": "pass" if all(item["status"] == "pass" for item in results) else "fail",
        "packages": results,
        "note": "Read-only integrity verification only; this command does not rerun analyses or access controlled raw inputs.",
    }
    print(json.dumps(output, ensure_ascii=True, indent=2))
    return 0 if output["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())

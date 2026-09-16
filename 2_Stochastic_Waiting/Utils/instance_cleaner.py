from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple


TARGET_META_KEYS = (
	"arrival_time_wave_changed",
	"arrival_wave_params",
	"original_arrival_times",
)


def iter_instance_json_files(source_dir: Path) -> Iterable[Path]:
	"""Yield all JSON files that contain 'instance' in the filename."""
	for path in source_dir.rglob("*.json"):
		if "instance" in path.name.lower():
			yield path


def clean_instance_file(file_path: Path) -> Tuple[bool, str]:
	"""
	Remove configured keys from file['meta'].

	Returns:
		(changed, status)
	"""
	try:
		with file_path.open("r", encoding="utf-8") as handle:
			content = json.load(handle)
	except (OSError, json.JSONDecodeError) as exc:
		return False, f"error: {exc}"

	if not isinstance(content, dict):
		return False, "skip: root is not a dict"

	meta = content.get("meta")
	if not isinstance(meta, dict):
		return False, "skip: no meta dict"

	removed_any = False
	for key in TARGET_META_KEYS:
		if key in meta:
			meta.pop(key, None)
			removed_any = True

	if not removed_any:
		return False, "skip: nothing to remove"

	try:
		with file_path.open("w", encoding="utf-8") as handle:
			json.dump(content, handle, ensure_ascii=False, indent=4)
			handle.write("\n")
	except OSError as exc:
		return False, f"error: {exc}"

	return True, "updated"


def clean_folder(source_dir: Path) -> Dict[str, int]:
	"""Process all matching files and print per-file status."""
	stats = {
		"found": 0,
		"updated": 0,
		"skipped": 0,
		"errors": 0,
	}

	for file_path in iter_instance_json_files(source_dir):
		stats["found"] += 1
		changed, status = clean_instance_file(file_path)

		if status.startswith("error"):
			stats["errors"] += 1
		elif changed:
			stats["updated"] += 1
		else:
			stats["skipped"] += 1

		print(f"[{status}] {file_path}")

	return stats


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Rekursiv JSON-Dateien mit 'instance' im Namen durchsuchen und "
			"Meta-Keys entfernen."
		)
	)
	parser.add_argument(
		"source",
		nargs="?",
		help="Pfad zum Quellordner. Falls leer, wird interaktiv abgefragt.",
	)
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	source_input = args.source or input("Quellordner eingeben: ").strip()

	source_dir = Path(source_input).expanduser().resolve()
	if not source_dir.is_dir():
		print(f"Ungueltiger Ordner: {source_dir}")
		return 1

	stats = clean_folder(source_dir)
	print(
		"\nFertig: "
		f"gefunden={stats['found']}, "
		f"aktualisiert={stats['updated']}, "
		f"uebersprungen={stats['skipped']}, "
		f"fehler={stats['errors']}"
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())


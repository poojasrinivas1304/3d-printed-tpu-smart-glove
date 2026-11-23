#!/usr/bin/env python3
"""Interactive launcher for the four-section glove study protocol.

This file keeps participant identification, section selection, model training,
and model selection in one place.  The existing four acquisition programs are
loaded as engines and are not modified.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import ModuleType


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "glove_data"
MODEL_DIR = BASE_DIR / "models"
CALIBRATION_MAX_AGE_HOURS = 4


@dataclass(frozen=True)
class Section:
    number: str
    title: str
    engine_file: str
    task: str
    mode: str
    session: str


SECTIONS = {
    "1": Section("1", "Finger calibration", "1_finger_collect.py", "finger", "calibrate", "FINGER_CAL"),
    "2": Section("2", "Finger blind test", "2_finger_blind.py", "finger", "blind", "FINGER_BLIND"),
    "3": Section("3", "Posture calibration", "3_posture_collect.py", "posture", "calibrate", "POSTURE_CAL"),
    "4": Section("4", "Posture blind test", "4_posture_blind.py", "posture", "blind", "POSTURE_BLIND"),
}

_MODULE_CACHE: dict[str, ModuleType] = {}


def load_engine(filename: str) -> ModuleType:
    """Load one of the existing numbered programs without launching its wrapper."""
    if filename in _MODULE_CACHE:
        return _MODULE_CACHE[filename]
    path = BASE_DIR / filename
    if not path.is_file():
        raise SystemExit(f"Required program not found: {path}")
    module_name = f"glove_engine_{path.stem.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load required program: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MODULE_CACHE[filename] = module
    return module


def run_engine(section: Section, argv: list[str], allowed_modes: list[str]) -> None:
    """Run an engine while preserving the acquisition file in saved provenance."""
    engine = load_engine(section.engine_file)
    original_argv0 = sys.argv[0]
    try:
        sys.argv[0] = str(BASE_DIR / section.engine_file)
        engine.main(argv=argv, fixed_task=section.task, allowed_modes=allowed_modes)
    finally:
        sys.argv[0] = original_argv0


def normalize_participant_id(value: str) -> str:
    """Return a de-identified participant label such as P01 or P02."""
    value = value.strip().replace(" ", "-").replace("_", "-")
    match = re.fullmatch(r"[Pp]?(\d{1,2})(?:-([A-Za-z][A-Za-z0-9-]{0,30}))?", value)
    if not match:
        raise ValueError("Use a de-identified label from P01 through P99.")
    number = int(match.group(1))
    if not 1 <= number <= 99:
        raise ValueError("Participant number must be between 1 and 99.")
    label = f"P{number:02d}"
    if match.group(2):
        label += "-" + match.group(2).lower()
    return label


def prompt_participant() -> tuple[str, str]:
    while True:
        try:
            participant = normalize_participant_id(
                input("Participant ID (for example P03): ")
            )
            break
        except ValueError as exc:
            print(exc)
    display_name = input(
        "Participant name (optional; shown now but not stored in research files): "
    ).strip()
    return participant, display_name


def prompt_fit() -> str:
    while True:
        fit = input("Glove fit (snug, normal, or loose): ").strip().lower()
        if fit in {"snug", "normal", "loose"}:
            return fit
        print("Please enter snug, normal, or loose.")


def prompt_donning() -> int:
    while True:
        entered = input(
            "Donning number (1 for the first fitting; increase after removing/refitting the glove): "
        ).strip()
        if entered.isdigit() and int(entered) >= 1:
            return int(entered)
        print("Please enter a whole number: 1, 2, 3, ...")


def yes_no(prompt: str, default: bool = False) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    answer = input(prompt + suffix).strip().lower()
    if not answer:
        return default
    return answer in {"y", "yes"}


def read_meta(folder: Path) -> dict | None:
    try:
        return json.loads((folder / "meta.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None


def completed_sessions(participant: str, section: Section) -> list[Path]:
    participant_dir = DATA_DIR / participant
    pattern = f"{participant}_{section.session}_{section.mode}_*"
    completed = []
    for folder in participant_dir.glob(pattern):
        meta = read_meta(folder)
        if meta and meta.get("status") == "completed":
            completed.append(folder)
    return sorted(completed, reverse=True)


def latest_completed_calibration(participant: str, task: str) -> Path | None:
    section = SECTIONS["1"] if task == "finger" else SECTIONS["3"]
    now = datetime.now()
    for folder in completed_sessions(participant, section):
        meta = read_meta(folder) or {}
        counts = meta.get("counts") or {}
        try:
            ended = datetime.fromisoformat(meta["ended_iso"])
            age_seconds = (now - ended).total_seconds()
        except (KeyError, TypeError, ValueError):
            continue
        if (
            meta.get("task") == task
            and counts.get("accepted_trials") == 50
            and (folder / "features.csv").is_file()
            and 0 <= age_seconds <= CALIBRATION_MAX_AGE_HOURS * 3600
        ):
            return folder
    return None


def warn_if_repeating(participant: str, section: Section) -> bool:
    previous = completed_sessions(participant, section)
    if not previous:
        return True
    print(f"\nA completed {section.title.lower()} already exists for {participant}:")
    print(f"  {previous[0]}")
    return yes_no("Record another completed session anyway?")


def run_calibration(
    participant: str,
    fit: str,
    donning: int,
    section: Section,
) -> Path | None:
    if not warn_if_repeating(participant, section):
        print("Section skipped.")
        return None

    before = set((DATA_DIR / participant).glob(f"{participant}_{section.session}_{section.mode}_*"))
    args = [
        "--mode", "calibrate",
        "--participant", participant,
        "--session", section.session,
        "--fit", fit,
        "--donning", str(donning),
        "--notes", (
            f"{participant} standardized study collection; {section.title.lower()}; "
            "left hand; seven sensors; dorsal glove arrangement; right hand excluded"
        ),
        "--out-dir", str(DATA_DIR),
        "--trials-per-class", "10",
        "--rest-every", "25",
        "--precond-cycles", "5",
    ]
    run_engine(section, args, ["calibrate"])

    after = set((DATA_DIR / participant).glob(f"{participant}_{section.session}_{section.mode}_*"))
    new_folders = sorted(after - before, reverse=True)
    for folder in new_folders:
        meta = read_meta(folder) or {}
        if meta.get("status") == "completed" and (meta.get("counts") or {}).get("accepted_trials") == 50:
            return folder
    print(f"\n{section.title} did not finish with 50 accepted trials. The next section will not start.")
    return None


def train_model_if_needed(section: Section, calibration: Path) -> Path:
    model_path = MODEL_DIR / f"{calibration.name}_v1.joblib"
    if model_path.is_file():
        print(f"\nUsing the existing frozen model:\n  {model_path}")
        return model_path

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nTraining and freezing the {section.task} model from:\n  {calibration}")
    run_engine(
        section,
        [
            "--mode", "train",
            "--input", str(calibration / "features.csv"),
            "--model-out", str(model_path),
            "--feature-set", "v1",
            "--cv-folds", "5",
            "--n-estimators", "700",
            "--model-seed", "42",
            "--min-prob", "0.35",
            "--margin", "0.08",
        ],
        ["train"],
    )
    if not model_path.is_file():
        raise SystemExit("Model training did not produce the expected model file.")
    return model_path


def run_blind(
    participant: str,
    fit: str,
    donning: int,
    section: Section,
    calibration: Path | None = None,
) -> bool:
    if not warn_if_repeating(participant, section):
        print("Section skipped.")
        return False

    calibration = calibration or latest_completed_calibration(participant, section.task)
    if calibration is None:
        print(
            f"\nNo completed 50-trial {section.task} calibration for {participant} was found "
            f"within the last {CALIBRATION_MAX_AGE_HOURS} hours."
        )
        print("Run the matching calibration section first, with the same participant ID.")
        return False

    model_path = train_model_if_needed(section, calibration)
    before = set((DATA_DIR / participant).glob(f"{participant}_{section.session}_{section.mode}_*"))
    run_engine(
        section,
        [
            "--mode", "blind",
            "--participant", participant,
            "--session", section.session,
            "--fit", fit,
            "--donning", str(donning),
            "--notes", (
                f"{participant} standardized study collection; {section.title.lower()}; "
                "left hand; seven sensors; dorsal glove arrangement; right hand excluded"
            ),
            "--out-dir", str(DATA_DIR),
            "--blind-trials", "10",
            "--rest-every", "0",
            "--precond-cycles", "0",
            "--model", str(model_path),
        ],
        ["blind"],
    )

    after = set((DATA_DIR / participant).glob(f"{participant}_{section.session}_{section.mode}_*"))
    new_folders = sorted(after - before, reverse=True)
    for folder in new_folders:
        if (read_meta(folder) or {}).get("status") == "completed":
            return True
    print(f"\n{section.title} did not complete successfully.")
    return False


def print_menu() -> None:
    print("\nSelect a section:")
    for number, section in SECTIONS.items():
        print(f"  {number}. {section.title}")
    print("  5. Run all four sections in sequence")
    print("  0. Exit")


def run_selected(participant: str, fit: str, donning: int, selection: str) -> None:
    if selection in {"1", "3"}:
        run_calibration(participant, fit, donning, SECTIONS[selection])
        return
    if selection in {"2", "4"}:
        run_blind(participant, fit, donning, SECTIONS[selection])
        return

    print("\nAll four sections will run in this order:")
    for section in SECTIONS.values():
        print(f"  {section.number}. {section.title}")
    print("Keep the glove and all seven sensors in the same position throughout.")
    print("Preconditioning runs in calibration sections 1 and 3 only; blind sections 2 and 4 skip it.")
    if not yes_no("Start the complete protocol?"):
        print("Cancelled.")
        return

    finger_cal = run_calibration(participant, fit, donning, SECTIONS["1"])
    if finger_cal is None:
        return
    if not run_blind(participant, fit, donning, SECTIONS["2"], finger_cal):
        return
    posture_cal = run_calibration(participant, fit, donning, SECTIONS["3"])
    if posture_cal is None:
        return
    run_blind(participant, fit, donning, SECTIONS["4"], posture_cal)


def main() -> None:
    print("=" * 68)
    print("GLOVE STUDY: UNIFIED DATA-COLLECTION LAUNCHER")
    print("Use anonymous participant IDs (P01, P02, ...), not names, in research files.")
    print("=" * 68)

    participant, display_name = prompt_participant()
    shown = f"{participant} ({display_name})" if display_name else participant
    print(f"\nSelected participant: {shown}")
    print("The optional name will not be written to the data or model files.")
    if not yes_no("Is this participant correct?"):
        print("Cancelled. Restart and enter the correct participant ID.")
        return

    print_menu()
    while True:
        selection = input("Section number: ").strip()
        if selection in {"0", "1", "2", "3", "4", "5"}:
            break
        print("Please select 0, 1, 2, 3, 4, or 5.")
    if selection == "0":
        print("No section selected.")
        return

    fit = prompt_fit()
    donning = prompt_donning()
    section_name = "All four sections" if selection == "5" else SECTIONS[selection].title
    print("\nPlease verify:")
    print(f"  Participant: {participant}")
    print(f"  Section:     {section_name}")
    print(f"  Glove fit:   {fit}")
    print(f"  Donning:     {donning}")
    print("  Hand:        left")
    print("  Sensors:     all seven")
    if not yes_no("Continue with this configuration?"):
        print("Cancelled.")
        return

    run_selected(participant, fit, donning, selection)
    print(f"\nFinished. Participant data folder:\n  {DATA_DIR / participant}")


if __name__ == "__main__":
    main()

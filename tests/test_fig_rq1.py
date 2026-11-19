"""scripts/fig_rq1.py must still draw when the Latin Modern font files are absent."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "fig_rq1.py"
RUNS = REPO_ROOT / "results" / "step2" / "runs" / "sentences_allagree"


class TestFigRq1FontFallback(unittest.TestCase):
    def test_runs_without_latin_modern_files(self) -> None:
        if not RUNS.is_dir():
            self.skipTest(f"Step 2 runs missing at {RUNS}")
        with tempfile.TemporaryDirectory() as tmp:
            font_dir = os.path.join(tmp, "no_fonts")
            mpl_dir = os.path.join(tmp, "mplconfig")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(font_dir)
            os.makedirs(mpl_dir)
            env = dict(os.environ)
            env["LM_FONT_DIR"] = font_dir
            env["MPLCONFIGDIR"] = mpl_dir
            env["MPLBACKEND"] = "Agg"
            completed = subprocess.run(
                [sys.executable, str(SCRIPT), str(REPO_ROOT), out_dir],
                env=env,
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

            figures = [
                path for path in Path(out_dir).iterdir()
                if path.suffix in (".pdf", ".png") and path.stat().st_size > 0
            ]
            self.assertGreaterEqual(len(figures), 1, os.listdir(out_dir))
            self.assertEqual(
                sorted(path.name for path in Path(out_dir).iterdir()),
                ["rq1_neutral_errors.pdf", "rq1_neutral_errors.png"],
            )

            # The fallback warning is printed once, or not at all when the machine has
            # Latin Modern Roman installed as a system font.
            warnings = [
                line for line in completed.stderr.splitlines()
                if "Latin Modern Roman not found" in line
            ]
            self.assertLessEqual(len(warnings), 1, completed.stderr)
            for line in warnings:
                self.assertIn(font_dir, line)
                self.assertIn("serif", line)


if __name__ == "__main__":
    unittest.main()

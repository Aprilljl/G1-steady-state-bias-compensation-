# Historical analysis unit tests

These are targeted adaptations of tests from the private source tree: `thesis_eval/tests/test_sp_design.py` and `thesis_eval/analysis/tests/test_sp_s1_analysis.py`, `test_sp_round2_analysis.py`, and `test_sp_e1_rule.py`. Imports now target `historical.thesis_eval.analysis`; the design end-to-end fixture uses a cleaned-up temporary directory. Test expressions were reformatted; E1's onset index is calculated once rather than once per cycle. No experiment criteria, analysis windows, or model selection are changed. Other original tests are not claimed to be ported. See `PROVENANCE.md`, `THIRD_PARTY_NOTICES.md`, and `historical/LICENSE` for the source and license notices; their existence is not a grant to publish this release.

From the project root, with NumPy and pytest installed (pytest is a test dependency and is not listed in `requirements.txt`):

```sh
python -B -m pytest -p no:cacheprovider tests/historical_analysis -v
```

The inputs are **synthetic**, not historical experimental measurements: in-memory arrays and generated rows, plus one generated CSV/NPZ fixture in a temporary directory. The latter contains 500 hazard rows, 30 static rows, and one synthetic step trace. The original speed-table boundary example checks 134.0200005 seconds of total exposure, including actual survivor exposure; the step fixture checks a fitted time constant near 0.5 seconds. These checks exercise code behavior and known examples, not the manuscript's historical numbers, simulation, learned policies, or scientific validity. No checkpoint or simulator is loaded; the end-to-end output is temporary. Some design tests compute simple mathematical plant recurrences, not a robot simulation. Running this suite does not modify historical inputs. The E1 rule's synthetic classification is not evidence that its post-hoc rule was preregistered or that periodic-orbit preconditions passed.

The suite includes pytest function tests and unittest classes; use pytest for the complete suite. `unittest discover` alone does not collect pytest functions and must not be reported as full coverage. No dependency installation or network access is performed by the command.

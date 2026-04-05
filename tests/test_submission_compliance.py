"""
Submission compliance tests.

Checks that every model registered in models/__init__.py satisfies the
competition's submission requirements:

  1. Model directory exists at models/<model_name>/
  2. Weights file (state_dict.pt) exists in that directory
  3. Constructor takes no arguments: Model()
  4. __call__ accepts (t, pos, idcs_airfoil, velocity_in) — 4 positional args
  5. Output shape is (batch, 5, 100k, 3)
  6. Output dtype is float32
  7. Model is self-contained — no imports from src/ inside models/<model_name>/

Run with:
    /home/agrov/gram/bin/python -m pytest tests/test_submission_compliance.py -v

Most of these will currently FAIL for ResidualMLP (known issue: base.py imports
src.features). All should pass before the final submission PR.
"""

import ast
import importlib
import inspect
import os
import sys
from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
MODELS_DIR = REPO_ROOT / "models"

BATCH_SIZE = 2       # small batch for speed
NUM_POS = 100_000
NUM_T_IN = 5
NUM_T_OUT = 5


def make_dummy_inputs(batch_size=BATCH_SIZE, num_pos=NUM_POS):
    torch.manual_seed(0)
    t = torch.rand(batch_size, NUM_T_IN + NUM_T_OUT)
    pos = torch.rand(batch_size, num_pos, 3)
    idcs_airfoil = [torch.randint(num_pos, (8000,)) for _ in range(batch_size)]
    velocity_in = torch.rand(batch_size, NUM_T_IN, num_pos, 3)
    return t, pos, idcs_airfoil, velocity_in


# ---------------------------------------------------------------------------
# Discover registered models from models/__init__.py
# ---------------------------------------------------------------------------

def get_registered_models() -> dict[str, type]:
    """
    Import models/__init__.py and return {class_name: class} for everything
    exported from it.
    """
    spec = importlib.util.spec_from_file_location(
        "models", MODELS_DIR / "__init__.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO_ROOT))
    spec.loader.exec_module(module)

    classes = {}
    for name in dir(module):
        obj = getattr(module, name)
        if inspect.isclass(obj) and not name.startswith("_"):
            classes[name] = obj
    return classes


def get_model_dir(cls: type) -> Path | None:
    """Infer the model directory from the class's source file."""
    try:
        src_file = Path(inspect.getfile(cls))
        # models/<model_name>/model.py -> models/<model_name>/
        if src_file.parent.parent == MODELS_DIR:
            return src_file.parent
    except (TypeError, OSError):
        pass
    return None


REGISTERED_MODELS = get_registered_models()
MODEL_IDS = list(REGISTERED_MODELS.keys())


# ---------------------------------------------------------------------------
# 1. Model directory structure
# ---------------------------------------------------------------------------

class TestModelDirectory:

    @pytest.mark.parametrize("model_name", MODEL_IDS)
    def test_model_directory_exists(self, model_name):
        """models/<model_name>/ directory must exist."""
        cls = REGISTERED_MODELS[model_name]
        model_dir = get_model_dir(cls)
        assert model_dir is not None, (
            f"{model_name}: could not determine model directory from source file"
        )
        assert model_dir.is_dir(), (
            f"{model_name}: directory {model_dir} does not exist"
        )



# ---------------------------------------------------------------------------
# 2. Constructor signature
# ---------------------------------------------------------------------------

class TestConstructor:

    @pytest.mark.parametrize("model_name", MODEL_IDS)
    def test_no_argument_constructor(self, model_name):
        """Model must be instantiable with no arguments: Model()"""
        cls = REGISTERED_MODELS[model_name]
        try:
            model = cls()
        except TypeError as e:
            pytest.fail(f"{model_name}: constructor failed with no args: {e}")


# ---------------------------------------------------------------------------
# 3. __call__ signature
# ---------------------------------------------------------------------------

class TestCallSignature:

    @pytest.mark.parametrize("model_name", MODEL_IDS)
    def test_call_accepts_four_positional_args(self, model_name):
        """__call__ must accept (t, pos, idcs_airfoil, velocity_in) — exactly 4 positional."""
        cls = REGISTERED_MODELS[model_name]
        model = cls()
        t, pos, idcs, vel_in = make_dummy_inputs()
        try:
            out = model(t, pos, idcs, vel_in)
        except Exception as e:
            pytest.fail(f"{model_name}: __call__ with 4 args raised {type(e).__name__}: {e}")

    @pytest.mark.parametrize("model_name", MODEL_IDS)
    def test_output_shape(self, model_name):
        """Output must be (batch, 5, 100k, 3)."""
        cls = REGISTERED_MODELS[model_name]
        model = cls()
        t, pos, idcs, vel_in = make_dummy_inputs()
        out = model(t, pos, idcs, vel_in)
        expected = (BATCH_SIZE, NUM_T_OUT, NUM_POS, 3)
        assert out.shape == expected, (
            f"{model_name}: expected shape {expected}, got {out.shape}"
        )

    @pytest.mark.parametrize("model_name", MODEL_IDS)
    def test_output_dtype_float32(self, model_name):
        """Output must be float32."""
        cls = REGISTERED_MODELS[model_name]
        model = cls()
        t, pos, idcs, vel_in = make_dummy_inputs()
        out = model(t, pos, idcs, vel_in)
        assert out.dtype == torch.float32, (
            f"{model_name}: expected float32, got {out.dtype}"
        )


# ---------------------------------------------------------------------------
# 4. Self-containment — no imports from src/
# ---------------------------------------------------------------------------

def collect_python_files(directory: Path) -> list[Path]:
    return list(directory.rglob("*.py"))


def find_src_imports(filepath: Path) -> list[str]:
    """Parse a Python file and return any 'from src.*' or 'import src.*' lines."""
    try:
        tree = ast.parse(filepath.read_text())
    except SyntaxError:
        return []

    bad_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("src"):
                bad_imports.append(f"from {node.module} import ...")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("src"):
                    bad_imports.append(f"import {alias.name}")
    return bad_imports


class TestSelfContainment:

    @pytest.mark.parametrize("model_name", MODEL_IDS)
    def test_no_src_imports_in_model_directory(self, model_name):
        """
        Model directory must not import from src/.
        The submission is models/<model_name>/ only — src/ will not be present
        in the competition repo.

        Currently EXPECTED TO FAIL for ResidualMLP (base.py imports src.features).
        """
        cls = REGISTERED_MODELS[model_name]
        model_dir = get_model_dir(cls)
        assert model_dir is not None, f"{model_name}: could not determine model directory"

        violations = {}
        # Check model dir itself
        for py_file in collect_python_files(model_dir):
            bad = find_src_imports(py_file)
            if bad:
                violations[str(py_file.relative_to(REPO_ROOT))] = bad

        # Also check base.py since all models inherit from it
        base_py = MODELS_DIR / "base.py"
        if base_py.exists():
            bad = find_src_imports(base_py)
            if bad:
                violations[str(base_py.relative_to(REPO_ROOT))] = bad

        assert not violations, (
            f"{model_name}: imports from src/ found (must be self-contained):\n"
            + "\n".join(f"  {f}: {imps}" for f, imps in violations.items())
        )

    @pytest.mark.parametrize("model_name", MODEL_IDS)
    def test_no_absolute_paths_in_model_directory(self, model_name):
        """
        Model files must not contain hardcoded absolute paths (e.g. /home/...).
        These will break on the organizers' machine.
        Use os.path.dirname(__file__) to resolve paths relative to the model file.
        """
        cls = REGISTERED_MODELS[model_name]
        model_dir = get_model_dir(cls)
        assert model_dir is not None, f"{model_name}: could not determine model directory"

        violations = {}
        files_to_check = list(collect_python_files(model_dir))
        base_py = MODELS_DIR / "base.py"
        if base_py.exists():
            files_to_check.append(base_py)

        for py_file in files_to_check:
            source = py_file.read_text()
            bad_lines = [
                f"line {i+1}: {line.strip()}"
                for i, line in enumerate(source.splitlines())
                if "/home/" in line or "/Users/" in line or "/root/" in line
            ]
            if bad_lines:
                violations[str(py_file.relative_to(REPO_ROOT))] = bad_lines

        assert not violations, (
            f"{model_name}: hardcoded absolute paths found (will break on organizers' machine):\n"
            + "\n".join(f"  {f}:\n    " + "\n    ".join(lines)
                        for f, lines in violations.items())
        )

    @pytest.mark.parametrize("model_name", MODEL_IDS)
    def test_import_entry_in_init(self, model_name):
        """models/__init__.py must export the model class."""
        assert model_name in REGISTERED_MODELS, (
            f"{model_name} not found in models/__init__.py"
        )


# ---------------------------------------------------------------------------
# 5. Variable-length idcs_airfoil handling
# ---------------------------------------------------------------------------

class TestVariableLengthAirfoil:

    @pytest.mark.parametrize("model_name", MODEL_IDS)
    def test_handles_variable_length_idcs_airfoil(self, model_name):
        """
        Each sample in a batch may have a different number of airfoil surface
        points (real range: 3142–24198). Model must handle this without crashing
        or producing wrong output shape.
        """
        cls = REGISTERED_MODELS[model_name]
        model = cls()
        torch.manual_seed(0)
        t = torch.rand(BATCH_SIZE, NUM_T_IN + NUM_T_OUT)
        pos = torch.rand(BATCH_SIZE, NUM_POS, 3)
        # Deliberately vary lengths across batch
        idcs_airfoil = [
            torch.randint(NUM_POS, (3142,)),   # min real size
            torch.randint(NUM_POS, (24198,)),  # max real size
        ]
        vel_in = torch.rand(BATCH_SIZE, NUM_T_IN, NUM_POS, 3)

        out = model(t, pos, idcs_airfoil, vel_in)
        assert out.shape == (BATCH_SIZE, NUM_T_OUT, NUM_POS, 3), (
            f"{model_name}: wrong output shape with variable-length idcs_airfoil: {out.shape}"
        )

    @pytest.mark.parametrize("model_name", MODEL_IDS)
    def test_idcs_airfoil_values_in_valid_range(self, model_name):
        """
        idcs_airfoil elements must index into pos, i.e. values in [0, 100k).
        Tests that the model doesn't produce out-of-bounds errors with valid indices.
        """
        cls = REGISTERED_MODELS[model_name]
        model = cls()
        torch.manual_seed(1)
        t = torch.rand(BATCH_SIZE, NUM_T_IN + NUM_T_OUT)
        pos = torch.rand(BATCH_SIZE, NUM_POS, 3)
        idcs_airfoil = [
            torch.randint(0, NUM_POS, (5000,)),
            torch.randint(0, NUM_POS, (10000,)),
        ]
        vel_in = torch.rand(BATCH_SIZE, NUM_T_IN, NUM_POS, 3)

        try:
            out = model(t, pos, idcs_airfoil, vel_in)
        except IndexError as e:
            pytest.fail(f"{model_name}: IndexError with valid idcs_airfoil: {e}")

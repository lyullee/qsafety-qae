import pytest

from qsafety.ibm import IBMPreview, run_ibm


def test_qpu_submission_requires_explicit_confirmation():
    preview = IBMPreview("unused", (), (), 1, ())
    with pytest.raises(PermissionError, match="confirm=True"):
        run_ibm(preview)

"""Save an IBM Quantum API token without placing it in shell history."""

from getpass import getpass

from qiskit_ibm_runtime import QiskitRuntimeService


token = getpass("IBM Quantum API token (input hidden): ")
QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform",
    token=token,
    plans_preference=["open"],
    region="us-east",
    set_as_default=True,
    overwrite=True,
)
print("IBM Quantum Open Plan account saved (region=us-east)")

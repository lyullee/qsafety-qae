from qsafety import RiskDistribution, compare_estimators
from qsafety.circuits import validate_bundle


risk = RiskDistribution.from_csv(
    "examples/phmsa_risk_strata.csv",
    label_column="label",
)
mc, qae = compare_estimators(risk, shots=512, repeats=200, seed=42)
print(f"Exact probability: {risk.probability:.6f}")
print(f"MC RMSE: {mc.rmse:.6g}")
print(f"Ideal ML-QAE RMSE: {qae.rmse:.6g}")
print(validate_bundle(risk))

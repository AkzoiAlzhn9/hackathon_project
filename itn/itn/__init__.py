"""Inverse text normalisation: BIO span tagging for Kazakh/Russian speech transcripts."""

CLASSES = (
    "CARDINAL",
    "ORDINAL",
    "DECIMAL",
    "DATE",
    "TIME",
    "MEASURE",
    "EMAIL",
    "WHITELIST",
)

LABELS = ("O",) + tuple(
    f"{prefix}-{cls}" for cls in CLASSES for prefix in ("B", "I")
)

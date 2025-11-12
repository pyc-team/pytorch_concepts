"""
Environment setup and warning filters.

This module should be imported first to configure warnings and environment settings.
"""
import warnings

# ============================================================================
# SUPPRESS THIRD-PARTY LIBRARY WARNINGS
# ============================================================================

# Suppress WandB's Pydantic v2 compatibility warnings
# These warnings come from WandB v0.22.2 internal code using Field(repr=False) 
# and Field(frozen=True) in a way incompatible with Pydantic v2's stricter rules.
# This is a known issue in WandB and does not affect functionality.
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pydantic._internal._generate_schema",
    message=".*'repr' attribute.*Field\\(\\).*"
)

warnings.filterwarnings(
    "ignore", 
    category=UserWarning,
    module="pydantic._internal._generate_schema",
    message=".*'frozen' attribute.*Field\\(\\).*"
)

# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================

# You can add other environment setup here if needed

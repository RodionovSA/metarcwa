# metarcwa/model/utils.py
# Backward-compat re-export shim. This module used to be a grab-bag holding
# dtype helpers, nn.Module registration helpers, and external-package
# adapters (ANALYSIS.md A2); those now live in their own modules
# (`metarcwa._dtypes`, `.nn_helpers`, `.adapters`). Kept here purely so
# existing `from metarcwa.model.utils import ...` importers keep working —
# new code should import directly from the modules below.

from .._dtypes import _REAL_TO_COMPLEX, to_complex, to_real   # noqa: F401
from .nn_helpers import register, CallableModule              # noqa: F401
from .adapters import from_metashapes, from_dispertorch        # noqa: F401

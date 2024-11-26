# __init__.py
import os
import glob
import importlib

# Get all Python files in the directory except __init__.py
module_files = glob.glob(os.path.join(os.path.dirname(__file__), "*.py"))
module_names = [os.path.basename(f)[:-3] for f in module_files if not f.endswith('__init__.py')]

# Import all modules and populate __all__
__all__ = []
for module_name in module_names:
    module = importlib.import_module(f".{module_name}", package=__name__)
    # Add module attributes to __all__
    if hasattr(module, '__all__'):
        __all__.extend(module.__all__)
    else:
        # Include all non-private attributes (not starting with _)
        __all__.extend(attr for attr in dir(module) if not attr.startswith("_"))

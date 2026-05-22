"""
setup_env.py — One-time environment setup for scoring pipeline
==============================================================
Run this ONCE at the top of your notebook (or as the first cell).
Installs only what's missing without upgrading existing packages.

Safe to re-run — skips packages that are already installed.

Usage (notebook cell):
    %run setup_env.py
"""

import subprocess
import sys

def _install_if_missing(package, import_name=None):
    """Install a package only if it can't be imported."""
    name = import_name or package
    try:
        __import__(name)
    except ImportError:
        print(f"   Installing {package}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package, "--no-deps", "--quiet"],
            stderr=subprocess.DEVNULL,
        )
        print(f"   Installed {package}")

_install_if_missing("pandas_ta", "pandas_ta")
_install_if_missing("polygon-api-client", "polygon")

# Verify critical dependencies exist (don't install — these come with the runtime)
_required = ['tensorflow', 'pandas', 'numpy', 'pyarrow', 'pytz', 'google.cloud.storage']
_missing = []
for mod in _required:
    try:
        __import__(mod)
    except ImportError:
        _missing.append(mod)

if _missing:
    print(f"\n   MISSING RUNTIME PACKAGES: {_missing}")
    print(f"   These should be pre-installed. Create a fresh runtime if needed.")
    sys.exit(1)
else:
    print("   Environment OK")

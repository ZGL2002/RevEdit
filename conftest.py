import sys
from pathlib import Path

REVEDIT_ROOT = Path(__file__).resolve().parent
BADEDIT_ROOT = REVEDIT_ROOT.parent
for p in (str(REVEDIT_ROOT), str(BADEDIT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

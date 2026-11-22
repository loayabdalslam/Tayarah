"""
Tayarah Terminal Entrypoint (Backward Compatibility Bridge)
Redirects cleanly to tayarah.terminal.main()
"""
import os
import sys

# Ensure package is on sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from tayarah.terminal import TayarahDualTerminal, main

if __name__ == "__main__":
    main()

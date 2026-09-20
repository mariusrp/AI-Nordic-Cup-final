"""evo_pe: patience4 (ripe_mode 2) + elder economy combined."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evo_common
Hivemind = evo_common.make(patience=1, ripe_mode=2, elder=1)

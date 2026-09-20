"""evo_patience2: motion-aware new-fruit detector (ripe_mode 1), ripe_s 18, min_frac 0.35."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evo_common
Hivemind = evo_common.make(patience=1, ripe_mode=1)

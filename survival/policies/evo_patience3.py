"""evo_patience3: motion-aware detector, shorter wait (ripe_s 14) and only agents above half energy wait (min_frac 0.5)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evo_common
Hivemind = evo_common.make(patience=1, ripe_mode=1, ripe_s=14.0, ripe_min_frac=0.5)

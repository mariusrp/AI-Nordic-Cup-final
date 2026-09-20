"""evo_elder: bb_juke + evo_common mechanism elder (see evo_common.py). Params via EVO_PARAMS or make(**kw)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evo_common
Hivemind = evo_common.make(elder=1)

"""Time the serving onset path (decode_audio on a temp .mp3 + spans.energy_onsets) per conversation."""
import glob
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spans as S  # noqa: E402
from faster_whisper.audio import decode_audio  # noqa: E402

aud = sorted(glob.glob(os.path.join(os.environ.get('UPSTREAM', '/workspace/upstream'),
                                    'medical-appointment/data/audio/*.mp3')))[: int(sys.argv[1]) if len(sys.argv) > 1 else 5]
for p in aud:
    b = open(p, 'rb').read()
    t = time.time()
    with tempfile.NamedTemporaryFile(suffix='.mp3') as f:
        f.write(b)
        f.flush()
        pcm = decode_audio(f.name, sampling_rate=16000)
    t1 = time.time()
    ons = S.energy_onsets(pcm)
    t2 = time.time()
    print(os.path.basename(p), f'{len(pcm) / 16000:.0f}s audio  decode {t1 - t:.2f}s  onsets {t2 - t1:.2f}s  n={len(ons)}', flush=True)

"""Pack everything except model checkpoints, the request cache and the key file into one archive to send back."""
import sys,tarfile,time
from pathlib import Path
out=Path(sys.argv[1]);name=out/f"results_{time.strftime('%Y%m%d_%H%M')}.tgz"
skip_dirs={'checkpoints','milestones','api_cache'}
def keep(info):
    parts=Path(info.name).parts
    if any(p in skip_dirs or p.startswith('global_step_') for p in parts):return None
    if parts[-1] in ('openai.env',) or parts[-1].startswith('results_') or parts[-1].endswith(('.safetensors','.pt','.bin')):return None
    return info
with tarfile.open(name,'w:gz') as tar:tar.add(out,arcname='out',filter=keep)
print(f'wrote {name} ({name.stat().st_size/2**20:.1f} MiB). It contains logs, trajectories, judge records, metrics and the API ledger; no checkpoints and no key.')

"""CPU/API work only. Never starts Router training or approves reviewed code."""
import sys,json,traceback,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'harnessaudit_verl'))
from guidance_rl.cli import main
out=Path(__file__).resolve().parents[1]
config=out/'control/launch.json'
# Independent artifacts share the same locked $10 budget and exact-request cache.
commands=('optimize',) if '--only-optimize' in sys.argv else ('controlled','optimize')
for command in commands:
    print('Starting',command,time.ctime(),flush=True)
    sys.argv=['guidance_rl',command,'--config',str(config)]
    try:main()
    except Exception as e:
        (out/'audit'/f'{command}_failure.json').write_text(json.dumps({'error_type':type(e).__name__,'message':str(e),'time':time.time()},indent=2))
        traceback.print_exc();raise
    if command=='optimize':
        from update_review import update
        update(out)
    print('Finished',command,time.ctime(),flush=True)

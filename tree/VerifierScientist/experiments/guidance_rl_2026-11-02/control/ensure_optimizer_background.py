"""Detach one bounded continuation; optimizer lock and budget prevent duplicate calls."""
import json,os,subprocess,sys,time
from pathlib import Path
out=Path(__file__).resolve().parents[1]
log=(out/'logs/optimizer_background.log').open('a')
p=subprocess.Popen([sys.executable,str(out/'control/run_cpu_experiments.py'),'--only-optimize'],cwd=str(out.parents[2]),
    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
log.close()
(out/'control/optimizer_background.json').write_text(json.dumps({'pid':p.pid,'started':time.time(),'max_usd':10,'note':'Same optimizer lock, cache and budget as foreground; completed rounds return from cache.'},indent=2))
print('Detached bounded optimizer continuation:',p.pid)

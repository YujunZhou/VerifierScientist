"""Progress summary of the smoke and full runs. Read-only."""
import json,sys,time
from pathlib import Path
out=Path(sys.argv[1])
led=out/'api_budget.json'
if led.exists():
    d=json.loads(led.read_text());print(f"API ledger: {d['spent']:.2f} USD over {d['calls']} calls; open reservations {len(d.get('reserved',{}))}")
for name in ('smoke_run','run'):
    run=out/name
    if not run.exists():print(f'{name}: not started');continue
    print(f'== {name}')
    plan=run/'execution_plan.json'
    stages=[s['stage'] for s in json.loads(plan.read_text())['stages']] if plan.exists() else []
    for s in stages:
        done=(run/s/'complete.json').exists();started=(run/s).exists()
        print(f"   {s:16s} {'done' if done else ('running or failed' if started else 'pending')}")
    for f in ('latest_complete.json','complete.json'):
        if (run/f).exists():print('  ',f,json.dumps(json.loads((run/f).read_text()))[:300])
    logs=sorted(run.glob('*/attempt_*/driver.log'),key=lambda p:p.stat().st_mtime)
    if logs:
        age=time.time()-logs[-1].stat().st_mtime;print(f'   newest log: {logs[-1]} (last written {age/60:.0f} min ago)')
        print('   '+'\n   '.join(logs[-1].read_text(errors='replace').splitlines()[-5:]))

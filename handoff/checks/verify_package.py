"""Step 3. Package integrity: every file the launcher hashes is present and identical to what was reviewed.
No GPU, no API calls. Takes about two minutes (it reads ~40,000 files)."""
import json,sys
from pathlib import Path
from guidance_rl.cli import review_files,preflight
from guidance_rl.pipeline import approval_digest
ok=True
args=sys.argv[1:]
for config,approval in zip(args[0::2],args[1::2]):
    c=json.loads(Path(config).read_text());a=json.loads(Path(approval).read_text())
    files,inputs=review_files(c)
    missing=sorted(set(c['code_files']+c['review_inputs'])-set(files+inputs));extra=sorted(set(files+inputs)-set(c['code_files']+c['review_inputs']))
    if missing or extra:
        ok=False;print(f'FAIL {Path(config).name}: file inventory differs; missing {len(missing)}, unexpected {len(extra)}')
        for p in missing[:10]:print('   missing   ',p)
        for p in extra[:10]:print('   unexpected',p)
        continue
    report=preflight(c)
    digest=approval_digest(c)
    same=digest==a.get('manifest')
    print(f"{Path(config).name}: {len(files)} code files, {len(inputs)} inputs, {len(report['required_fixtures'])} task repositories ok; digest {'matches' if same else 'DIFFERS FROM'} the approval")
    if not same:
        ok=False;print('   computed',digest,'approved',a.get('manifest'))
print('VERIFY','PASSED' if ok else 'FAILED');sys.exit(0 if ok else 1)

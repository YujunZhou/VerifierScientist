"""Step 4. Can this machine reach the OpenAI API with the provided key, through the project's own client?
Makes ONE tiny real call to the judge model (cost well under 0.01 USD, recorded in the run's ledger)."""
import json,sys,time
from pathlib import Path
from pydantic import BaseModel
from guidance_rl.client import OpenAIClient
c=json.loads(Path(sys.argv[1]).read_text());api=c['api'];model=c['optimization']['judge_model']
key=Path(api['env_file'])
if not key.is_file():
    print(f'FAIL: key file not found. Create <OUT>/openai.env containing one line:  OPENAI_API_KEY=...');sys.exit(1)
class Pong(BaseModel):
    answer:int
client=OpenAIClient(api);t=time.time()
try:
    out=client.call('Reply with the requested integer.','What is 17 + 25?',Pong,model=model,tag='handoff:netcheck',salt=f'netcheck-{int(t)}',max_output_tokens=2048)
except Exception as exc:
    print(f'FAIL: {type(exc).__name__}: {str(exc)[:600]}')
    print('If this is a connection error, the machine cannot reach', api['base_url'], '- set HTTPS_PROXY before run.sh if a proxy is required, and report the exact error.')
    sys.exit(1)
ledger=json.loads(Path(api['budget_file']).read_text())
print(f"model {model} answered {out.answer} in {time.time()-t:.1f}s; ledger now {ledger['spent']:.4f} USD over {ledger['calls']} calls (cap {api['max_usd']} USD)")
ok=out.answer==42
print('NETCHECK','PASSED' if ok else 'FAILED');sys.exit(0 if ok else 1)

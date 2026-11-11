"""Step 5. Start the local inference server exactly as the training does (on the 4th GPU), ask it one
question, stop it. No API calls. Takes 3-10 minutes; the first start compiles GPU kernels."""
import json,os,socket,subprocess,sys,time,urllib.request
from pathlib import Path
from guidance_rl.training import runtime_env,stop_owned
from guidance_rl.pipeline import assigned_gpus
c=json.loads(Path(sys.argv[1]).read_text());gpus=assigned_gpus(os.environ);env=runtime_env(c)
out=Path(c['run_dir']).parent/'worker_check';out.mkdir(parents=True,exist_ok=True)
with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
with socket.socket() as s:s.bind(('127.0.0.1',0));grpc=s.getsockname()[1]
url=f'http://127.0.0.1:{port}';wenv=dict(env,CUDA_VISIBLE_DEVICES=gpus[-1],SGLANG_GRPC_PORT=str(grpc))
cmd=[c['python'],'-m','sglang.launch_server','--model-path',c['model_path'],'--served-model-name',c['model_path'],
     '--host','127.0.0.1','--port',str(port),'--tp','1','--context-length','32768','--mem-fraction-static','0.8',
     '--tool-call-parser','qwen3_coder','--reasoning-parser','qwen3','--disable-cuda-graph','--trust-remote-code']
log=out/'worker.log';print('starting inference server on GPU',gpus[-1],'- log:',log,flush=True)
ok=False;proc=None;t=time.time()
try:
    with log.open('a') as f:
        proc=subprocess.Popen(cmd,env=wenv,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        deadline=time.monotonic()+900
        while True:
            if proc.poll() is not None:raise RuntimeError('server exited during start-up')
            try:
                with urllib.request.urlopen(url+'/health_generate',timeout=60) as r:
                    if r.status==200:break
            except OSError:pass
            if time.monotonic()>deadline:raise TimeoutError('server not healthy after 15 minutes')
            time.sleep(5)
        print(f'server healthy after {time.time()-t:.0f}s',flush=True)
        body=json.dumps({'model':c['model_path'],'messages':[{'role':'user','content':'Reply with the single word: ready'}],'max_tokens':256,'temperature':0}).encode()
        req=urllib.request.Request(url+'/v1/chat/completions',data=body,headers={'Content-Type':'application/json','Authorization':'Bearer EMPTY'})
        with urllib.request.urlopen(req,timeout=300) as r:reply=json.loads(r.read())
        msg=reply['choices'][0]['message'];print('model reply:',repr((msg.get('content') or '')[:200]),'| tokens',reply.get('usage',{}).get('completion_tokens'))
        ok=True
except Exception as exc:
    print(f'FAIL: {type(exc).__name__}: {exc}');print('--- last 60 lines of the server log ---')
    print('\n'.join(log.read_text(errors='replace').splitlines()[-60:]))
finally:stop_owned(proc)
print('WORKERCHECK','PASSED' if ok else 'FAILED');sys.exit(0 if ok else 1)

"""Step 1. GPU and driver check. No API calls, no cost."""
import json,subprocess,sys
ok=True
def fail(msg):
    global ok;ok=False;print('FAIL:',msg)
try:
    raw=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total,driver_version','--format=csv,noheader,nounits'],check=True,text=True,capture_output=True,timeout=60).stdout.strip()
except Exception as exc:
    print('FAIL: nvidia-smi is not usable inside the container:',exc);sys.exit(1)
rows=[[x.strip() for x in line.split(',')] for line in raw.splitlines()]
print('GPUs visible in the container:')
for r in rows:print('  ',r)
if len(rows)!=4:fail(f'the run needs exactly 4 GPUs in the container, found {len(rows)} (set GPUS=a,b,c,d for run.sh)')
for r in rows:
    if 'A100' not in r[1] and 'H100' not in r[1]:fail(f'GPU {r[0]} is "{r[1]}"; the code accepts only A100 or H100')
    if int(float(r[2]))<70000:print(f'WARNING: GPU {r[0]} has {r[2]} MiB. The run was only ever tested on 80 GB cards; 40 GB cards may run out of memory. Report this number.')
driver=rows[0][3] if rows else '0'
print('driver version:',driver)
try:
    if int(driver.split('.')[0])<580:fail(f'driver {driver} is older than 580. The environment is built for CUDA 13.0 (PyTorch 2.11 cu130), which needs driver 580 or newer.')
except ValueError:print('WARNING: could not parse driver version')
import torch
print('torch',torch.__version__,'built for CUDA',torch.version.cuda)
if not torch.cuda.is_available():fail('torch.cuda.is_available() is False')
else:
    n=torch.cuda.device_count();print('torch sees',n,'devices')
    if n!=4:fail('torch does not see 4 devices')
    for i in range(n):
        x=torch.randn(4096,4096,device=f'cuda:{i}',dtype=torch.bfloat16);y=(x@x).float().abs().mean().item()
        free,total=torch.cuda.mem_get_info(i);print(f'   cuda:{i} {torch.cuda.get_device_name(i)} matmul ok ({y:.1f}); free {free/2**30:.1f} of {total/2**30:.1f} GiB')
        if free<0.9*total:print(f'WARNING: cuda:{i} is already in use by another process ({(total-free)/2**30:.1f} GiB taken). The run needs the four cards to itself.')
print('GPUCHECK','PASSED' if ok else 'FAILED');sys.exit(0 if ok else 1)

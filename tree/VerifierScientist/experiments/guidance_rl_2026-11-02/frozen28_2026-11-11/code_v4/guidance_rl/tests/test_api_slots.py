"""Real process tests; no model, API, Ray or GPU dependencies."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

AVAILABLE = importlib.util.find_spec('guidance_rl.api_slots') is not None
if AVAILABLE:
    from guidance_rl.api_slots import bounded_api_slot


CHILD = '''
import json,os,sys,time
from guidance_rl.api_slots import bounded_api_slot
config=json.loads(sys.argv[1])
with bounded_api_slot(config):
    print(json.dumps({'event':'enter','time':time.monotonic(),'pid':os.getpid()}),flush=True)
    time.sleep(float(sys.argv[2]))
    if len(sys.argv)>3: os._exit(13)
    print(json.dumps({'event':'leave','time':time.monotonic(),'pid':os.getpid()}),flush=True)
'''


class APISlotTests(unittest.TestCase):
    def test_module_exists(self):
        self.assertTrue(AVAILABLE, 'cross-process API slots are not implemented')

    @unittest.skipUnless(AVAILABLE, 'implementation pending')
    def test_six_real_processes_never_exceed_two_slots(self):
        with tempfile.TemporaryDirectory() as d:
            cfg={'budget_file':str(Path(d)/'budget.json'),'concurrency':2,'slot_wait_timeout':10}
            children=[subprocess.Popen([sys.executable,'-B','-u','-c',CHILD,json.dumps(cfg),'.2'],
                                      stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for _ in range(6)]
            events=[]
            try:
                for child in children:
                    out,err=child.communicate(timeout=15)
                    self.assertEqual(child.returncode,0,err)
                    events.extend(json.loads(line) for line in out.splitlines())
            finally:
                for child in children:
                    if child.poll() is None:child.kill();child.wait()
            active=maximum=0
            for event in sorted(events,key=lambda x:x['time']):
                active+=1 if event['event']=='enter' else -1
                maximum=max(maximum,active)
                self.assertGreaterEqual(active,0)
                self.assertLessEqual(active,2)
            self.assertEqual(active,0)
            self.assertEqual(maximum,2)
            self.assertEqual(len(events),12)
            self.assertEqual(len(list(Path(str(cfg['budget_file'])+'.slots').glob('*.lock'))),2)

    @unittest.skipUnless(AVAILABLE, 'implementation pending')
    def test_exception_releases_slot_for_another_process_and_keeps_inode(self):
        with tempfile.TemporaryDirectory() as d:
            cfg={'budget_file':str(Path(d)/'budget.json'),'concurrency':1,'slot_wait_timeout':2}
            with self.assertRaisesRegex(RuntimeError,'example'):
                with bounded_api_slot(cfg):raise RuntimeError('example')
            lock=next(Path(str(cfg['budget_file'])+'.slots').glob('*.lock'));inode=lock.stat().st_ino
            p=subprocess.run([sys.executable,'-B','-u','-c',CHILD,json.dumps(cfg),'0'],capture_output=True,text=True,timeout=5)
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertEqual(lock.stat().st_ino,inode)

    @unittest.skipUnless(AVAILABLE, 'implementation pending')
    def test_timeout_does_not_steal_live_lock(self):
        with tempfile.TemporaryDirectory() as d:
            cfg={'budget_file':str(Path(d)/'budget.json'),'concurrency':1,'slot_wait_timeout':.15}
            with bounded_api_slot(cfg):
                start=time.monotonic()
                with self.assertRaises(TimeoutError):
                    with bounded_api_slot(cfg):self.fail('live slot was stolen')
                self.assertGreaterEqual(time.monotonic()-start,.12)
                self.assertLess(time.monotonic()-start,1)
            with bounded_api_slot(cfg):pass

    @unittest.skipUnless(AVAILABLE, 'implementation pending')
    def test_process_crash_releases_os_lock(self):
        with tempfile.TemporaryDirectory() as d:
            cfg={'budget_file':str(Path(d)/'budget.json'),'concurrency':1,'slot_wait_timeout':2}
            p=subprocess.run([sys.executable,'-B','-u','-c',CHILD,json.dumps(cfg),'0','crash'],capture_output=True,text=True,timeout=5)
            self.assertEqual(p.returncode,13,p.stderr)
            with bounded_api_slot(cfg):pass

    @unittest.skipUnless(AVAILABLE, 'implementation pending')
    def test_rejects_invalid_bounds(self):
        with tempfile.TemporaryDirectory() as d:
            for limits in ({'concurrency':0},{'concurrency':1.5},{'concurrency':True},
                           {'concurrency':2,'slot_wait_timeout':-1},{'concurrency':2,'slot_wait_timeout':float('inf')}):
                with self.subTest(limits=limits),self.assertRaises(ValueError):
                    with bounded_api_slot({'budget_file':str(Path(d)/'budget.json'),**limits}):pass


if __name__=='__main__':unittest.main(verbosity=2)

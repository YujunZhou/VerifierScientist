import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from unittest.mock import Mock

from guidance_rl import allocation
from guidance_rl.core import atomic_json


class TaskApprovalTests(unittest.TestCase):
    def run_case(self, change=None, hold=False, approve=True):
        self.assertTrue(hasattr(allocation, 'execute_task'), 'Per-task approval enforcement missing')
        with TemporaryDirectory() as directory:
            root=Path(directory); marker=root/'executed'; source=root/'source.py'
            source.write_text('print("reviewed")\n')
            approval=root/'approval.json'; task_path=root/'task.json'; result=root/'result.json'
            task={'argv':[sys.executable,'-c',f'from pathlib import Path; p=Path({str(marker)!r}); p.write_text((p.read_text() if p.exists() else "")+"ran")'],
                  'cwd':str(root),'timeout_seconds':5,'operation':'train',
                  'input_files':[str(source)],'approval':str(approval)}
            if approve:
                atomic_json(approval,{'approved':True,'approved_by':'test fixture only','job_id':'test-job',
                    'operation':'train','task_fingerprint':allocation.task_fingerprint(task)})
            if change:change(task,source,approval)
            atomic_json(task_path,task)
            hold_path=root/'hold.json';atomic_json(hold_path,{'active':hold,'training_allowed':not hold})
            with patch.dict(os.environ,{'JOB_ID':'test-job'}):
                allocation.execute_task(task_path,result,root/'logs',{'research_hold':str(hold_path)})
                allocation.execute_task(task_path,result,root/'logs',{'research_hold':str(hold_path)})
            if marker.exists():self.assertEqual(marker.read_text(),'ran')
            return marker.exists(),json.loads(result.read_text())

    def test_missing_approval_never_executes(self):
        exists,result=self.run_case(approve=False)
        self.assertFalse(exists);self.assertEqual(result['status'],'rejected')

    def test_changed_command_never_executes(self):
        exists,result=self.run_case(lambda t,s,a:t['argv'].append('changed'))
        self.assertFalse(exists);self.assertEqual(result['status'],'rejected')

    def test_changed_source_never_executes(self):
        exists,result=self.run_case(lambda t,s,a:s.write_text('unreviewed'))
        self.assertFalse(exists);self.assertEqual(result['status'],'rejected')

    def test_training_hold_overrides_prior_task_approval(self):
        exists,result=self.run_case(hold=True)
        self.assertFalse(exists);self.assertEqual(result['status'],'rejected')

    def test_approved_command_executes_once(self):
        exists,result=self.run_case()
        self.assertTrue(exists);self.assertEqual(result['returncode'],0)

    def test_wrong_allocation_rejected(self):
        def change(task,source,approval):
            a=json.loads(approval.read_text());a['job_id']='different';atomic_json(approval,a)
        exists,result=self.run_case(change)
        self.assertFalse(exists);self.assertEqual(result['status'],'rejected')

    def test_interrupt_cleans_owned_process_group(self):
        process=Mock(pid=12345)
        process.wait.side_effect=[KeyboardInterrupt(),0]
        process.poll.return_value=None
        with TemporaryDirectory() as directory:
            root=Path(directory);task=root/'task.json'
            atomic_json(task,{'argv':['fake-reviewed-command'],'cwd':str(root),'timeout_seconds':5})
            with patch.object(allocation,'require_task_approval'),patch.object(allocation.subprocess,'Popen',return_value=process),patch.object(allocation.os,'killpg') as kill:
                with self.assertRaises(KeyboardInterrupt):allocation.execute_task(task,root/'result.json',root/'logs',{})
                kill.assert_called_once_with(12345,allocation.signal.SIGTERM)


if __name__=='__main__':unittest.main()

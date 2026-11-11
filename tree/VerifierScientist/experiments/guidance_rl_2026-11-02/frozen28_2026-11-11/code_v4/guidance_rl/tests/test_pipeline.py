import importlib.util
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
import json

EXISTS=importlib.util.find_spec('guidance_rl.pipeline') is not None
if EXISTS:
    from guidance_rl.pipeline import extract_final, select_candidate, approval_digest, require_approval, assigned_gpus

class PipelineTests(unittest.TestCase):
    def test_module_implemented(self): self.assertTrue(EXISTS,'pipeline not implemented')
    @unittest.skipUnless(EXISTS,'module pending')
    def test_final_excludes_reasoning_and_preserves_visible_text(self):
        self.assertEqual(extract_final('<think>secret</think>Answer<|im_end|>','final')['text'],'Answer')
        self.assertEqual(extract_final('hidden reasoning</think>Answer','final')['text'],'Answer')
        self.assertEqual(extract_final('<think>unfinished','length')['text'],'')
        self.assertEqual(extract_final('Partial answer','length')['text'],'Partial answer')
    @unittest.skipUnless(EXISTS,'module pending')
    def test_prompt_open_reasoning_is_not_visible(self):
        self.assertEqual(extract_final('private thoughts','length',thinking_open=True)['text'],'')
    @unittest.skipUnless(EXISTS,'module pending')
    def test_tool_xml_is_not_user_answer(self):
        self.assertEqual(extract_final('<tool_call><function=x>','final')['status'],'malformed_tool')
        self.assertEqual(extract_final('<tool_call>x</tool_call>','final')['text'],'')
    @unittest.skipUnless(EXISTS,'module pending')
    def test_selection_rejects_recall_hack(self):
        old={'pair_acc':.6,'macro_f1':.6,'clean_false_alarm':.1,'per_code':{'V-ID':{'gold':2,'recall':.5}}}
        bad={**old,'pair_acc':.9,'per_code':{'V-ID':{'gold':2,'recall':0}}}
        good={**old,'pair_acc':.7}
        self.assertEqual(select_candidate({'old':old,'hack':bad,'good':good},'old'),'good')
    @unittest.skipUnless(EXISTS,'module pending')
    def test_approval_binds_config_and_code(self):
        with TemporaryDirectory() as d:
            code=Path(d)/'code.py';code.write_text('a=1');conf={'code_files':[str(code)],'x':1}
            a=Path(d)/'approval.json';a.write_text(json.dumps({'approved_by':'human','manifest':approval_digest(conf)}))
            require_approval(conf,a)
            code.write_text('a=2')
            with self.assertRaises(ValueError):require_approval(conf,a)
    @unittest.skipUnless(EXISTS,'module pending')
    def test_gpu_mapping_never_discovers_extra_devices(self):
        self.assertEqual(assigned_gpus({'CUDA_VISIBLE_DEVICES':'2,3,5,7'}),['2','3','5','7'])
        with self.assertRaises(ValueError):assigned_gpus({})
        with self.assertRaises(ValueError):assigned_gpus({'CUDA_VISIBLE_DEVICES':'0,1'})

if __name__=='__main__':unittest.main()

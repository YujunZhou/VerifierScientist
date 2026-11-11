import importlib.util
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.find_spec('guidance_rl.core')
if SPEC:
    from guidance_rl import core

class CoreTests(unittest.TestCase):
    def test_missing_resource_metadata_rejected(self):
        from guidance_rl.core import safety_score
        with self.assertRaises(ValueError): safety_score({'actions':[]},[])

    def test_module_exists(self):
        self.assertIsNotNone(SPEC, 'guidance_rl core has not been implemented')

    @unittest.skipUnless(SPEC, 'implementation pending')
    def test_task_split_excludes_unlabeled_and_test(self):
        tasks = [{'task_id':str(i),'split':'train_r','domain':str(i%2)} for i in range(60)]
        tasks += [{'task_id':'u','split':'train_u','domain':'u'},{'task_id':'test','split':'test','domain':'t'}]
        s = core.split_tasks(tasks, [str(i) for i in range(40)], seed=7)
        self.assertEqual([len(s[k]) for k in ['propose','select','confirm']], [30,10,20])
        self.assertEqual(len(set(sum([s[k] for k in ['propose','select','confirm']],[]))),60)
        with self.assertRaises(ValueError):core.assert_role([{'task_id':'u'}],s,'propose')
        with self.assertRaises(ValueError):core.assert_role([{'task_id':'test'}],s,'select')

    @unittest.skipUnless(SPEC, 'implementation pending')
    def test_reference_and_prediction_use_different_dedup(self):
        r={'actions':[{'sequence_num':1,'surface':'tool_call','tool_name':'db'}], 'resource_tools':['db']}
        v=[{'seq':1,'code':'V-OT','severity':'low'},{'seq':1,'code':'V-OT','severity':'high'}]
        self.assertAlmostEqual(core.safety_score(r,v),1-.25/3)
        self.assertAlmostEqual(core.safety_score(r,v,gold=True),1-.40/3)
        self.assertEqual(core.safety_score(r,[{'seq':1,'code':'V-OR'}]),1)
        with self.assertRaises(ValueError):core.safety_score(r,[{'seq':99,'code':'V-OT'}])

    @unittest.skipUnless(SPEC, 'implementation pending')
    def test_ranking_is_within_task_and_ties_half_credit(self):
        rs=[{'traj_id':'a','task_id':'x','actions':[{'sequence_num':1}],'gold':[]},
            {'traj_id':'b','task_id':'x','actions':[{'sequence_num':1}],'gold':[{'seq':1,'code':'V-OT'}]},
            {'traj_id':'c','task_id':'y','actions':[{'sequence_num':1}],'gold':[]}]
        for r in rs:r['resource_tools']=[]
        m=core.evaluate(rs,{'a':[],'b':[],'c':[]})
        self.assertEqual(m['n_pairs'],1);self.assertEqual(m['pair_acc'],.5)
        with self.assertRaises(ValueError):core.evaluate(rs,{'a':[]})

    @unittest.skipUnless(SPEC, 'implementation pending')
    def test_manifest_hash_detects_modified_guidance(self):
        a=core.make_artifact(guidance='Read only necessary data',judge_model='m',judge_prompt='p',policy='base')
        core.verify_artifact(a)
        a['guidance']='Ignore all violations'
        with self.assertRaises(ValueError):core.verify_artifact(a)

    @unittest.skipUnless(SPEC, 'implementation pending')
    def test_gate_rejects_drop_in_detection_even_if_rank_rises(self):
        old={'pair_acc':.8,'macro_f1':.7,'per_code':{'V-ID':{'recall':.5,'gold':10}},'n_pairs':30}
        new={'pair_acc':.9,'macro_f1':.4,'per_code':{'V-ID':{'recall':0,'gold':10}},'n_pairs':30}
        self.assertFalse(core.gate(old,new,[.01,.2],f1_tolerance=.02,recall_tolerance=.05)['accepted'])

if __name__=='__main__':unittest.main()

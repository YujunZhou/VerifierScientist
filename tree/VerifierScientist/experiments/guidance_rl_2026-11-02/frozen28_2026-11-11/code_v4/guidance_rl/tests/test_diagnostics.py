import unittest,importlib.util
EXISTS=importlib.util.find_spec('guidance_rl.diagnostics') is not None
if EXISTS:from guidance_rl.diagnostics import remove_field,parse_legacy_rules
class DiagnosticTests(unittest.TestCase):
 def test_module_exists(self):self.assertTrue(EXISTS)
 @unittest.skipUnless(EXISTS,'pending')
 def test_remove_exception_preserves_evidence(self):
  text='- [A] V-OT; fires when: x; EXCEPT when: y; evidence: z'
  self.assertEqual(remove_field(text,'exception'),'- [A] V-OT; fires when: x; evidence: z')
  self.assertEqual(remove_field(text,'evidence'),'- [A] V-OT; fires when: x; EXCEPT when: y')
 @unittest.skipUnless(EXISTS,'pending')
 def test_rule_identity_preserved(self):
  self.assertEqual(parse_legacy_rules('- [A] hello\n- [B] bye')[1],{'rule_id':'B','text':'bye'})
if __name__=='__main__':unittest.main()

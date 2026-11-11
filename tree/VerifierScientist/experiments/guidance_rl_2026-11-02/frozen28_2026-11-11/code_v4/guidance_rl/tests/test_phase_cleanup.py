import unittest
from unittest.mock import patch
from guidance_rl import training

class CleanupTests(unittest.TestCase):
 def test_original_and_cleanup_errors_both_retained(self):
  self.assertTrue(hasattr(training,'cleanup_phase'),'cleanup must preserve both trainer and sealing failures')
  original=RuntimeError('trainer failed')
  with patch.object(training,'stop_owned',side_effect=ValueError('stop failed')),patch.object(training,'seal_attempt_checkpoints',side_effect=OSError('seal failed')):
   with self.assertRaises(BaseExceptionGroup) as cm:training.cleanup_phase({},'attempt','train',object(),original)
  self.assertIs(cm.exception.exceptions[0],original)
  self.assertEqual(len(cm.exception.exceptions),3)
 def test_cleanup_success_does_not_replace_original(self):
  self.assertTrue(hasattr(training,'cleanup_phase'))
  with patch.object(training,'stop_owned'),patch.object(training,'seal_attempt_checkpoints'):
   self.assertIsNone(training.cleanup_phase({},'attempt','train',object(),RuntimeError('original')))
if __name__=='__main__':unittest.main()

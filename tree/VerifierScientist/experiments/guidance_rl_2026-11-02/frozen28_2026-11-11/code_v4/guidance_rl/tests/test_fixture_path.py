import os,unittest
from pathlib import Path
from unittest.mock import patch
from ha_verl.paths import fixtures_root,harnessaudit_root
class FixturePath(unittest.TestCase):
 def test_new_run_fixture_override_preserves_legacy_default(self):
  with patch.dict(os.environ,{'HA_FIXTURES_ROOT':'/tmp/new-fixtures'}):self.assertEqual(fixtures_root(),Path('/tmp/new-fixtures'))
  with patch.dict(os.environ,{},clear=True):self.assertEqual(fixtures_root(),harnessaudit_root()/'fixtures')
if __name__=='__main__':unittest.main()

"""Merge completed local runtime evidence; never fabricate readiness."""
from pathlib import Path
import fcntl
import json
import subprocess

EXP = Path(__file__).resolve().parents[1]


def main():
    path = EXP / 'fixtures/restore_manifest.json'
    with (EXP / 'audit/fixture_runtime/manifest.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = json.loads(path.read_text())
        for name in ('app-admin', 'app-ecommerce', 'psf-requests-1921'):
            evidence = EXP / 'audit/fixture_runtime_local' / name / 'runtime_result.json'
            if not evidence.exists():
                continue
            row = json.loads(evidence.read_text())
            if row['fixture'] != name:
                raise ValueError('Local evidence task mismatch')
            if row.get('ready'):
                head = subprocess.run(['git', '-C', str(EXP / 'fixtures' / name), 'rev-parse', 'HEAD'],
                                      check=True, capture_output=True, text=True, timeout=30).stdout.strip()
                if head != row.get('git_head'):
                    raise ValueError('Local fixture changed since validation')
            item = manifest['fixtures'][name]
            for key in ('ready', 'validation_complete', 'environment', 'git_head', 'git_tree', 'tracked_files_sha256'):
                if key in row:
                    item[key] = row[key]
            item['runtime_evidence'] = str(evidence)
            item['runtime_error'] = row.get('error')
        manifest['ready'] = all(manifest['fixtures'][name].get('ready') is True
                                for name in manifest['required_fixtures'])
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(manifest, indent=2)); temp.replace(path)
    print(json.dumps({'ready': manifest['ready'], 'validated': sum(
        manifest['fixtures'][name].get('ready') is True for name in manifest['required_fixtures']),
        'required': len(manifest['required_fixtures'])}), flush=True)


if __name__ == '__main__':
    main()

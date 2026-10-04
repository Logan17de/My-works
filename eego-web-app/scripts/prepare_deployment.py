"""Stage the canonical Edge source locally. This script does not deploy anything."""
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[1]
staging = root / 'supabase' / 'functions' / 'eego-api'
staging.mkdir(parents=True, exist_ok=True)
shutil.copyfile(root / 'backend' / 'index.ts', staging / 'index.ts')
(root / 'supabase' / 'config.toml').write_text(
    'project_id = "eego"\n\n'
    '# Custom Eego sessions and scoped worker tokens are verified by the handler.\n'
    '# This is not an anonymous data API. Do not remove handler authentication.\n'
    '[functions.eego-api]\nverify_jwt = false\n'
)
print('Local Supabase staging files prepared. No deployment was attempted.')

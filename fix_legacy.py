import os, glob

for filepath in glob.glob('smoke_legacy/*.py'):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    changed = False
    
    # Fix docstring splits
    if '\"\"\"\n\n\"\"\"' in content:
        content = content.replace('\"\"\"\n\n\"\"\"', '\n\n')
        changed = True

    # Fix legacy imports in these smoke tests
    to_fix = [
        ('from smoke_test_semantic_strong import', 'from smoke_legacy.smoke_test_semantic_strong import'),
        ('from smoke_test_team_state_sync import', 'from smoke_legacy.smoke_test_team_state_sync import'),
        ('from smoke_test_real_takeover_playball import', 'from smoke_legacy.smoke_test_real_takeover_playball import'),
        ('from smoke_test_real_env_semantic_variants import', 'from smoke_legacy.smoke_test_real_env_semantic_variants import'),
        ('from smoke_test_real_takeover_completion import', 'from smoke_legacy.smoke_test_real_takeover_completion import'),
        ('from smoke_test_real_takeover_env_positive import', 'from smoke_legacy.smoke_test_real_takeover_env_positive import')
    ]
    for old, new in to_fix:
        if old in content:
            content = content.replace(old, new)
            changed = True
            
    if changed:
        with open(filepath, 'w', encoding='utf-8', newline='\n') as f:
            f.write(content)
        print('Fixed: ' + filepath)

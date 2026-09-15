with open('README.md', 'r', encoding='utf-8') as f:
    content = f.read()

checks = [
    ('YouTube link', 'ugT-6m7i8ls'),
    ('Demo video section', 'Full Demo Video'),
    ('GIF grid', 'seed_0.gif'),
    ('Scoring table', 'Scoring Rubric'),
    ('Speechmatics section', 'Speechmatics'),
    ('Architecture diagram', 'mermaid'),
    ('Benchmark results', '85.34 FPS'),
    ('Pre-submission checklist', 'Pre-Submission'),
    ('Getting Started', 'Getting Started'),
    ('Docker instructions', 'docker build'),
    ('Tests badge 39', '39'),
    ('Task suite table', 'set_table'),
    ('Repo structure tree', 'sentineledge-core/'),
    ('Action filter mention', 'action_filter'),
    ('Foxglove mention', 'foxglove'),
    ('Wrist/camera mention', 'camera'),
    ('ROS2 mention', 'ROS'),
    ('Model card link', 'MODEL_CARD'),
]

print('README Section Audit:')
all_ok = True
for name, pattern in checks:
    found = pattern in content
    if not found:
        all_ok = False
    status = 'OK' if found else 'MISSING'
    print(f'  [{status}] {name}')

if all_ok:
    print('\nAll checks passed!')
else:
    print('\nSome sections missing - needs fix.')

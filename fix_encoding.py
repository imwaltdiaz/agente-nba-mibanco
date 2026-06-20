import pathlib

files = [
    'src/data_prep.py',
    'src/train_causal.py',
    'src/inference_causal.py',
]

replacements = [
    ('\u2713', '[OK]'),
    ('\u26a0', '[WARN]'),
    ('\u2192', '->'),
    ('\u2014', '-'),
    ('\u00d7', 'x'),
    ('\u2500', '-'),
    ('\u2550', '='),
    ('\u2116', 'No.'),
    ('\u00b0', 'deg'),
    ('\u2248', '~'),
    ('\u2260', '!='),
    ('\u2265', '>='),
    ('\u2264', '<='),
    ('\u2019', "'"),
    ('\u201c', '"'),
    ('\u201d', '"'),
]

ascii_encode_trick = ".encode('ascii', 'replace').decode('ascii')"

for fpath in files:
    p = pathlib.Path(fpath)
    text = p.read_text(encoding='utf-8')
    original_len = len(text)
    for old, new in replacements:
        text = text.replace(old, new)
    # Remove the encode trick leftover
    text = text.replace(ascii_encode_trick, '')
    p.write_text(text, encoding='utf-8')
    print(f'Fixed: {fpath} ({original_len} -> {len(text)} bytes)')

print('All scripts cleaned of non-ASCII chars.')

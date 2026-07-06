#!/usr/bin/env python3
"""Folio-RTF -> searchable corpus pipeline.

Usage:  python3 pipeline.py input.RTF output_prefix
Emits:  output_prefix.jsonl   {"ref":..., "text":...} per passage
        output_prefix_data.js same data for the offline search page

Folio marks each record with its reference as HIDDEN text ({\v BG 9.22});
we capture those as passage boundaries. Balaram font -> Unicode IAST.
Deterministic - no model needed.
"""
import sys, re, json

BAL = str.maketrans('äéüåèìïëñçöòàùÄÉÜÅÈÌÏËÑÇÖÒÀÙ',
                    'āīūṛṝḷñṇṣśṭḍṁḥĀĪŪṚṜḶÑṆṢŚṬḌṀḤ')

TOK = re.compile(r"\\([a-z]{1,32})(-?\d{1,10})?[ ]?|\\'([0-9a-fA-F]{2})|\\([^a-zA-Z])|([{}])")
SKIP_GROUPS = {'fonttbl','colortbl','stylesheet','info','pict','object','field'}
MARK = '\x01'

def rtf_to_text(path, out, blocksize=1 << 22):
    depth, skip_depth, hidden_depth = 0, None, None
    carry = ''
    hbuf = []
    def flush_hidden(buf):
        if hbuf:
            mk = ''.join(hbuf).strip()
            hbuf.clear()
            if 0 < len(mk) < 150:
                buf.append('\n' + MARK + mk + '\n')
    with open(path, 'r', encoding='latin-1') as f:
        while True:
            block = f.read(blocksize)
            data = carry + block
            if not data:
                break
            if block:
                cut = max(data.rfind(' '), data.rfind('}'), data.rfind('\n'))
                if cut <= 0:
                    carry = data
                    continue
                data, carry = data[:cut + 1], data[cut + 1:]
            else:
                carry = ''
            buf, i, n = [], 0, len(data)
            def emit(txt):
                t = txt.replace('\n', '').replace('\r', '')
                if skip_depth is not None:
                    return
                if hidden_depth is not None:
                    hbuf.append(t)
                else:
                    buf.append(t)
            while i < n:
                m = TOK.search(data, i)
                if not m:
                    emit(data[i:])
                    break
                if m.start() > i:
                    emit(data[i:m.start()])
                i = m.end()
                word, hexc, esc, brace = m.group(1), m.group(3), m.group(4), m.group(5)
                if brace == '{':
                    depth += 1
                elif brace == '}':
                    depth -= 1
                    if skip_depth is not None and depth < skip_depth:
                        skip_depth = None
                    if hidden_depth is not None and depth < hidden_depth:
                        hidden_depth = None
                        flush_hidden(buf)
                elif hexc is not None:
                    b = int(hexc, 16)
                    try:
                        emit(bytes([b]).decode('cp1252'))
                    except UnicodeDecodeError:
                        emit(chr(b))
                elif esc is not None:
                    if esc in '\\{}':
                        emit(esc)
                    elif esc == '*' and skip_depth is None:
                        skip_depth = depth
                    elif esc == '~':
                        emit(' ')
                elif word:
                    if word == 'v':
                        if m.group(2) == '0':
                            if hidden_depth is not None:
                                hidden_depth = None
                                flush_hidden(buf)
                        elif hidden_depth is None and skip_depth is None:
                            hidden_depth = depth
                    elif word in SKIP_GROUPS:
                        if skip_depth is None:
                            skip_depth = depth
                    elif skip_depth is None and hidden_depth is None:
                        if word in ('par','line','sect','page'):
                            buf.append('\n')
                        elif word == 'tab':
                            buf.append(' ')
                        elif word == 'emdash':
                            buf.append('—')
                        elif word == 'endash':
                            buf.append('–')
                        elif word == 'u' and m.group(2):
                            buf.append(chr(int(m.group(2)) & 0xFFFF))
            out.write(''.join(buf))
            if not block:
                break

def is_gibberish(line):
    s = line.strip()
    if len(s) < 4 or s.startswith(MARK):
        return False
    letters = sum(ch.isalpha() for ch in s)
    if letters == 0:
        return False
    caps_mid = len(re.findall(r'[a-zā-ṣ][A-Z“”"&{}]|[A-Za-z][)(=+]', s))
    return caps_mid / max(letters, 1) > 0.12

# a hidden marker qualifies as a passage reference if it looks like a
# citation (book code + number, letter, lecture, etc.)
# v2 FIX: first char may be ANY letter (Ādi, Śrīmad-... were being rejected
# by the ASCII-only rule, which merged whole sections into giant records)
HASDIGIT = re.compile(r'\d')
SUBHEAD = re.compile(r'^(TEXTS?|CHAPTER|PURPORT|TRANSLATION|SYNONYMS|WORD-FOR-WORD|VERSE)\b', re.I)

TAPECODE = re.compile(r'^\d{6}[A-Za-z0-9.\-]*[A-Za-z][A-Za-z0-9.\-]*$')
FILECODE = re.compile(r'^[A-Za-z][A-Za-z0-9_\-]{1,30}\.[A-Z]{2,4}$')   # HUME.SYA etc.

def is_refmark(mk):
    if not mk or len(mk) > 141: return False
    if mk[0].isdigit():
        # tape codes like 660219-20BG.NY are record boundaries;
        # verse lines ('3: O my teacher...') are not
        return bool(TAPECODE.match(mk)) and len(mk) < 40
    if not mk[0].isalpha(): return False
    if SUBHEAD.match(mk): return False        # in-record headings, not refs
    if FILECODE.match(mk): return True        # v4: Philosophy Discussion records
    return bool(HASDIGIT.search(mk)) or mk.startswith('Letter')

def chunk(lines):
    ref, buf, n_since = None, [], 0
    for ln in lines:
        st = ln.strip()
        if not ln.startswith(MARK) and TAPECODE.match(st) and len(st) < 40:
            # visible tape-code line = start of a new record
            if any(x.strip() for x in buf):
                yield (ref or 'Front matter'), '\n'.join(buf).strip()
            ref, buf = st, []
            continue
        if ln.startswith(MARK):
            mk = ln[1:].strip()
            if is_refmark(mk):
                if mk == ref:        # duplicated marker for same record
                    continue
                if not any(x.strip() for x in buf):
                    # marker chain with no text between: keep the first
                    # (record) reference, ignore refinements
                    if ref is not None:
                        continue
                    ref = mk
                    continue
                yield (ref or 'Front matter'), '\n'.join(buf).strip()
                ref, buf = mk, []
            continue
        buf.append(ln)
    if any(x.strip() for x in buf):
        yield (ref or 'Front matter'), '\n'.join(buf).strip()

def main(src, prefix):
    tmp = prefix + '.rawtxt'
    with open(tmp, 'w', encoding='utf-8') as out:
        rtf_to_text(src, out)
    n = 0
    with open(tmp, encoding='utf-8') as f, \
         open(prefix + '.jsonl', 'w', encoding='utf-8') as jf, \
         open(prefix + '_data.js', 'w', encoding='utf-8') as sf:
        sf.write('window.CORPUS = window.CORPUS || [];\nwindow.CORPUS.push(...')
        recs = []
        kept = [l.translate(BAL) for l in f if not is_gibberish(l)]
        for ref, text in chunk(kept):
            text = re.sub(r'\n{3,}', '\n\n', text).strip()
            if len(text) < 2:
                continue
            rec = {'ref': ref, 'text': text}
            jf.write(json.dumps(rec, ensure_ascii=False) + '\n')
            recs.append(rec)
            n += 1
        sf.write(json.dumps(recs, ensure_ascii=False))
        sf.write(');\n')
    print(f'{n} passages -> {prefix}.jsonl / {prefix}_data.js')

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

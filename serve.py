#!/usr/bin/env python3
"""folio hybrid search app (v2) — fully offline, no LLM, quote-only.

Web UI:   python serve.py            -> http://localhost:7777
CLI:      python serve.py [--in BG,SB,Conversations] "query"

Searches every corpus listed in CORPORA. Keyword search always works;
semantic search activates for corpora whose index dir exists (built by
build_index_v2b.py). Results are verbatim quotes with references.
"""
import json, os, re, sys, unicodedata
import numpy as np
_TOOLDIR=os.path.dirname(os.path.abspath(__file__))
if os.path.isdir(os.path.join(_TOOLDIR,'models')) and os.listdir(os.path.join(_TOOLDIR,'models')):
    os.environ.setdefault('HF_HUB_OFFLINE','1'); os.environ.setdefault('TRANSFORMERS_OFFLINE','1')

TOOL = os.path.dirname(os.path.abspath(__file__))
CORPORA = {
    # name: (passages jsonl, semantic index dir or None)
    'folio': ('corpus/library.jsonl', 'index'),
}
TOPK = 12
_model = None

import re as _re
SECTIONS = ['BG','SB','CC','Other Books','Lectures','Conversations','Letters','Other','Notes']
_CONV = _re.compile(r'^(Room|Morning|Evening|Garden|Talk|Walk|Conversation|Interview|Press|Darśana|Darsana|Meeting|Discussion(?!s on)|Answers|Questions|Airport|Car |Train|VD \d)', _re.I)
_LECT = _re.compile(r'(Lecture|Address|Speech|Class$|Bhajan|Festival|Appearance Day|Disappearance Day|Initiation|Wedding|Cornerstone|Arrival|Departure|Purport to)', _re.I)
_LECT2 = _re.compile(r'^(Bhagavad-g[īi]t[āa]|Śr[īi]mad-Bh[āa]gavatam|Sri Isopanisad|Śr[īi] [ĪI]śopaniṣad|Brahma-saṁhitā|Caitanya-carit[āa]mṛta|Nectar of Devotion) .{0,40}--', _re.I)
_OB = _re.compile(r'^(KB|Kb|KRP|KṚṢṆA|NoD|TLC|Iso|EK|POP|SSR|MG|LOB|LoB|NBS|Mm|MM|TLK|TQK|PQPA|RV|PIE|NAM|SVA|SMD|BBD|MoG|OWK|Teachings|Kṛṣṇa Book|Path of|Perfect|Beyond|Easy Journey|Message of|Rāja-vidyā|Elevation|On the Way)\b')
_DATE = _re.compile(r'\b(19[5-7][0-9])\b')

def classify(ref, corpus):
    if corpus == 'notes': return 'Notes'
    if ref.startswith(('Bg ','Bg-','Bg.')) : return 'BG'
    if ref.startswith('SB'): return 'SB'
    if ref.startswith(('Ādi','Adi','Madhya','Antya','CC')): return 'CC'
    if ref.startswith('Letter'): return 'Letters'
    if _OB.search(ref): return 'Other Books'
    if _CONV.search(ref): return 'Conversations'
    if _LECT2.search(ref) or _LECT.search(ref): return 'Lectures'
    if _DATE.search(ref) and ',' in ref: return 'Conversations'
    return 'Other'

_NTBL = {}
def _nmap(ch):
    d = unicodedata.normalize('NFD', ch)
    return ''.join(c for c in d if not unicodedata.combining(c)).lower()
def norm(s):
    for ch in set(s):
        o = ord(ch)
        if o not in _NTBL:
            _NTBL[o] = _nmap(ch)
    return s.translate(_NTBL)

def _norm_with_map(s):
    """Return (normalized_string, orig2norm, norm2orig) index maps."""
    parts = []; o2n = []; n2o = []
    ni = 0
    for oi, ch in enumerate(s):
        nc = _nmap(ch)
        o2n.append(ni)
        for c in nc:
            n2o.append(oi)
            ni += 1
        parts.append(nc)
    o2n.append(ni)
    n2o.append(len(s))
    return ''.join(parts), o2n, n2o

def _highlight(body, qtext):
    """Highlight query terms in body, diacritic-insensitive. Returns HTML."""
    import re as _r, html as _h
    if not qtext.strip():
        return _h.escape(body)
    needles = [norm(x) for x in _r.findall(r'"([^"]+)"', qtext)]
    needles += [norm(x) for x in _r.sub(r'"[^"]*"', ' ', qtext).split() if x]
    if not needles:
        return _h.escape(body)
    nb, o2n, n2o = _norm_with_map(body)
    spans = []
    for t in needles:
        if not t: continue
        st = 0
        while True:
            p = nb.find(t, st)
            if p < 0: break
            orig_start = n2o[p]
            orig_end = n2o[min(p + len(t), len(n2o) - 1)]
            spans.append((orig_start, orig_end))
            st = p + 1
    if not spans:
        return _h.escape(body)
    spans.sort()
    merged = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    outp = []; last = 0
    for a, b in merged:
        outp.append(_h.escape(body[last:a]))
        outp.append('<mark>' + _h.escape(body[a:b]) + '</mark>')
        last = b
    outp.append(_h.escape(body[last:]))
    return ''.join(outp)

class Corpus:
    def __init__(self, name, jsonl, idxdir):
        self.name = name
        jp = os.path.join(TOOL, jsonl)
        self.recs = [json.loads(l) for l in open(jp,encoding='utf-8')]
        for r in self.recs: r['sec'] = classify(r['ref'], name)
        # Folio repeats some sections verbatim (e.g. Song Purports); keep first copy
        first = {}
        self.dup = {i for i, r in enumerate(self.recs)
                    if first.setdefault((r['ref'], r['text']), i) != i}
        cache = jp + '.norms.txt'
        self.norms = None
        if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(jp):
            lines = open(cache,encoding='utf-8').read().split('\n')
            if len(lines) >= len(self.recs):
                self.norms = lines[:len(self.recs)]
        if self.norms is None:
            self.norms = [norm((r['text']+' '+r['ref']).replace('\n',' ')) for r in self.recs]
            with open(cache,'w',encoding='utf-8') as cf:
                cf.write('\n'.join(self.norms))
        self.sem = None
        d = idxdir and os.path.join(TOOL, idxdir)
        if d and os.path.exists(os.path.join(d,'emb.f16.npy')):
            self.wins = [json.loads(l) for l in open(os.path.join(d,'windows.jsonl'),encoding='utf-8')]
            M = np.load(os.path.join(d,'emb.f16.npy')).astype(np.float32)
            M /= (np.linalg.norm(M,axis=1,keepdims=True)+1e-9)
            self.sem = M

    def skip(self, i, secs):
        """Duplicate record, or outside the selected sections."""
        return i in self.dup or bool(secs and self.recs[i]['sec'] not in secs)

    def keyword(self, q, secs=None):
        phrases = re.findall(r'"([^"]+)"', q)
        terms = [t for t in re.sub(r'"[^"]*"',' ',q).split() if t]
        needles = [norm(x) for x in (phrases+terms)]
        if not needles: return []
        hits=[]
        for i,n in enumerate(self.norms):
            if self.skip(i, secs): continue    # filter BEFORE the top-K cut
            if all(x in n for x in needles):
                sc = sum(min(n.count(x),4) for x in needles) + 2000/(len(n)+500)
                pos = min((n.find(x) for x in needles))
                hits.append((sc,i,pos))
        hits.sort(reverse=True)
        return hits[:TOPK]

    def semantic(self, qv, secs=None):
        if self.sem is None: return []
        sc = self.sem @ qv
        best={}
        # walk windows best-first until TOPK distinct allowed records are found
        for t in np.argsort(-sc):
            w=self.wins[t]; ri=w['ri']
            if self.skip(ri, secs): continue
            if ri not in best:
                if len(best) >= TOPK: break
                best[ri]=(float(sc[t]), t)
        out=[(s, ri, self.wins[t]) for ri,(s,t) in best.items()]
        out.sort(reverse=True)
        return out[:TOPK]

def get_model():
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        _model = TextEmbedding('BAAI/bge-small-en-v1.5',
                               cache_dir=os.path.join(TOOL,'models'),
                               threads=os.cpu_count())
    return _model

def embed_query(q):
    v = np.array(list(get_model().embed(
        ['Represent this sentence for searching relevant passages: '+q]))[0], dtype=np.float32)
    return v/np.linalg.norm(v)

def search(cs, q, mode='hybrid', secs=None):
    results=[]
    qv = None
    if mode in ('hybrid','semantic') and any(c.sem is not None for c in cs):
        try:
            qv = embed_query(q)
        except Exception as e:
            print('[semantic disabled: embedding model failed to load —', str(e)[:120], ']')
    # pre-compute normalized query terms for term-boost and supplementary pass
    _qneedles=[]
    if q.strip():
        _qneedles=[norm(x) for x in re.findall(r'"([^"]+)"', q)]
        _qneedles+=[norm(x) for x in re.sub(r'"[^"]*"',' ',q).split() if x]
    for c in cs:
        seen=set()
        if mode in ('hybrid','keyword'):
            for sc,i,pos in c.keyword(q, secs):
                seen.add(i)
                results.append({'corpus':c.name,'i':i,'ref':c.recs[i]['ref'],'sec':c.recs[i]['sec'],'score':round(sc,2),
                                'how':'keyword','text':c.recs[i]['text'],'snippet_at':pos or 0})
        if qv is not None and mode in ('hybrid','semantic'):
            for sc,i,w in c.semantic(qv, secs):
                if i in seen: continue
                seen.add(i)
                t=c.recs[i]['text']
                # boost semantic score if record contains actual query terms
                n_t=c.norms[i] if i<len(c.norms) else norm(t)
                term_hits=sum(1 for nd in _qneedles if nd and nd in n_t)
                boost = 0.15 * term_hits / max(len(_qneedles),1) if _qneedles else 0
                results.append({'corpus':c.name,'i':i,'ref':c.recs[i]['ref'],'sec':c.recs[i]['sec'],
                                'score':round(sc+boost,3),
                                'how':'semantic','text':t,
                                'snippet_at': w['a'] if w else 0})
        # supplement: find records containing most query terms that keyword/semantic missed
        if _qneedles and mode in ('hybrid','semantic'):
            for i,n in enumerate(c.norms):
                if i in seen or c.skip(i, secs): continue
                hits=sum(1 for nd in _qneedles if nd and nd in n)
                if hits >= max(len(_qneedles)-1, 1):
                    seen.add(i)
                    pos=min((n.find(nd) for nd in _qneedles if nd and nd in n), default=0)
                    results.append({'corpus':c.name,'i':i,'ref':c.recs[i]['ref'],'sec':c.recs[i]['sec'],
                                    'score':round(0.5 + 0.1*hits,3),
                                    'how':'keyword+','text':c.recs[i]['text'],'snippet_at':pos})
    results.sort(key=lambda r: -r['score'])
    return results

PAGE = open(os.path.join(TOOL,'serve_page.html'),encoding='utf-8').read() \
       if os.path.exists(os.path.join(TOOL,'serve_page.html')) else None

def main():
    cs = []
    for name,(jsonl,idx) in CORPORA.items():
        if os.path.exists(os.path.join(TOOL,jsonl)):
            print('loading', name, '...')
            cs.append(Corpus(name,jsonl,idx))
            print(' ', name, len(cs[-1].recs), 'passages',
                  '(semantic ON)' if cs[-1].sem is not None else '(keyword only)')
    args=sys.argv[1:]; secs=None
    if args and args[0]=='--in':
        secs=set(args[1].split(',')); args=args[2:]
    if args:
        q=' '.join(args)
        for r in search(cs,q,secs=secs)[:15]:
            at=r.get('snippet_at',0)
            print(f"\n[{r['how']} {r['score']}] {r.get('sec','')} :: {r['ref']}")
            print(r['text'][at:at+400].replace('\n',' '))
        return
    from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
    from urllib.parse import urlparse, parse_qs
    class H(BaseHTTPRequestHandler):
        def log_message(s,*a): pass
        def do_GET(s):
            u=urlparse(s.path)
            if u.path=='/api':
                qs=parse_qs(u.query); q=qs.get('q',[''])[0]; mode=qs.get('mode',['hybrid'])[0]
                sp=qs.get('secs',[''])[0]
                secs=set(sp.split(',')) if sp else None
                try:
                    out=search(cs,q,mode,secs) if q.strip() else []
                except Exception as e:
                    import traceback; traceback.print_exc()
                    s.send_response(500); s.end_headers(); s.wfile.write(str(e).encode()); return
                for r in out:
                    at=r.pop('snippet_at',0)
                    # For semantic results, try to find actual query terms and re-center snippet
                    if r.get('how')=='semantic' and q.strip():
                        import re as _r2
                        needles=[norm(x) for x in _r2.findall(r'"([^"]+)"', q)]
                        needles+=[norm(x) for x in _r2.sub(r'"[^"]*"',' ',q).split() if x]
                        nt=norm(r['text'])
                        for nd in needles:
                            if not nd: continue
                            p=nt.find(nd)
                            if p>=0:
                                _,_o2n,_n2o=_norm_with_map(r['text'])
                                at=_n2o[min(p,len(_n2o)-1)]
                                break
                    pre='…' if at>100 else ''
                    r['snippet']=pre+r['text'][max(0,at-100):at+500]
                    if len(r['text'])>6000:
                        a=max(0,at-2000)
                        r['text']=('…' if a>0 else '')+r['text'][a:at+4000]+'\n… [long record trimmed around the match]'
                    r['at_hint']=at
                body=json.dumps(out).encode()
                s.send_response(200); s.send_header('Content-Type','application/json'); s.end_headers()
                s.wfile.write(body)
            elif u.path=='/record':
                qs=parse_qs(u.query)
                cname=qs.get('corpus',[''])[0]; idx=int(qs.get('i',['-1'])[0])
                qtext=qs.get('q',[''])[0]
                cobj=next((c for c in cs if c.name==cname), None)
                if not cobj or not (0<=idx<len(cobj.recs)):
                    s.send_response(404); s.end_headers(); return
                r=cobj.recs[idx]
                body=r['text']
                body = _highlight(body, qtext)
                page=('<!DOCTYPE html><html><head><meta charset="utf-8"><title>'+r['ref']+
                    '</title><style>body{font-family:Georgia,serif;background:#faf6ef;color:#2b2216;'
                    'max-width:820px;margin:0 auto;padding:30px 16px;line-height:1.6}'
                    'h1{color:#8a5a2b;font-size:1.2em} mark{background:#f3d9a4}'
                    'pre{white-space:pre-wrap;font-family:inherit}</style></head><body>'
                    '<h1>'+r['ref']+'</h1><div style="color:#7a6a52">'+r.get('sec','')+' · '+cname+
                    ' · full record ('+str(len(r['text']))+' chars)</div><pre>'+body+'</pre></body></html>')
                s.send_response(200); s.send_header('Content-Type','text/html; charset=utf-8'); s.end_headers()
                s.wfile.write(page.encode())
            else:
                s.send_response(200); s.send_header('Content-Type','text/html; charset=utf-8'); s.end_headers()
                s.wfile.write((PAGE or '<h1>missing serve_page.html</h1>').encode())
    print('\nOpen http://localhost:7777  (Ctrl+C or close this window to stop)')
    srv = ThreadingHTTPServer(('127.0.0.1',7777), H)
    srv.daemon_threads = True
    try:
        import threading, webbrowser
        threading.Timer(1.0, lambda: webbrowser.open('http://localhost:7777')).start()
    except Exception:
        pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print('\nstopped.')

if __name__=='__main__':
    main()

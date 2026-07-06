#!/usr/bin/env python3
"""Semantic index builder for the corpus produced by pipeline.py.
Creates windows + texts if missing, then embeds resumably. Rerun until COMPLETE."""
import json, os, sys, time
import numpy as np
# work offline when the model is already cached locally; otherwise allow
# a one-time download of the small embedding model
if os.path.isdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),'models')) and \
   os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),'models')):
    os.environ.setdefault('HF_HUB_OFFLINE','1'); os.environ.setdefault('TRANSFORMERS_OFFLINE','1')

TOOL = os.path.dirname(os.path.abspath(__file__))
IDX  = os.path.join(TOOL, 'index')
CORPUS = os.path.join(TOOL, 'corpus', 'library.jsonl')
WFILE, TFILE = os.path.join(IDX,'windows.jsonl'), os.path.join(IDX,'wtexts.txt')

def make_texts():
    if os.path.exists(TFILE) and os.path.getsize(TFILE) > 100_000_000: return
    print('preparing window texts (one-time, a few minutes)...', flush=True)
    lib = open(CORPUS, encoding='utf-8')
    cur_ri, cur_text = -1, ''
    with open(WFILE, encoding='utf-8') as wf, open(TFILE,'w',encoding='utf-8') as tf:
        for l in wf:
            w = json.loads(l)
            while cur_ri < w['ri']:
                cur_text = json.loads(lib.readline())['text']; cur_ri += 1
            tf.write(cur_text[w['a']:w['b']].replace('\n',' ').replace('\r',' ') + '\n')
    print('texts done')

def embed(limit_s):
    print('importing fastembed...', flush=True)
    from fastembed import TextEmbedding
    t0 = time.time()
    total = sum(1 for _ in open(WFILE, encoding='utf-8'))
    print(f'{total} windows in index plan; scanning existing checkpoints...', flush=True)
    parts = sorted(f for f in os.listdir(IDX) if f.startswith('emb_part'))
    done = 0
    for p in parts:
        try:
            done += np.load(os.path.join(IDX,p), mmap_mode='r').shape[0]
        except Exception:
            print('corrupt checkpoint removed:', p, flush=True)
            os.remove(os.path.join(IDX,p))
    if done >= total:
        M = np.vstack([np.load(os.path.join(IDX,p)).astype(np.float16) for p in parts])
        np.save(os.path.join(IDX,'emb.f16.npy'), M)
        print('merged', M.shape); return True
    print(f'resuming at {done}/{total} windows — loading embedding model...', flush=True)
    model = TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir=os.path.join(TOOL,'models'), threads=os.cpu_count())
    print('model loaded — embedding begins now (first checkpoint in ~3 min)', flush=True)
    out, i = [], 0
    B = 32
    CHECKPOINT_S = 180           # save + report every ~3 minutes
    last_save = t0
    saved = done
    def flush():
        nonlocal out, saved, last_save
        if not out: return
        M = np.vstack(out); out = []
        parts_now = sorted(f for f in os.listdir(IDX) if f.startswith('emb_part'))
        np.save(os.path.join(IDX, f'emb_part{len(parts_now):04d}.npy'), M)
        saved += M.shape[0]; last_save = time.time()
        print(f'checkpoint: {saved}/{total} windows embedded '
              f'({round(100*saved/total,1)}%) at {round(time.time()-t0)}s', flush=True)
    batch = []
    with open(TFILE, encoding='utf-8') as tf:
        for i, line in enumerate(tf):
            if i < done: continue
            batch.append(line.rstrip('\n')[:6000])
            if len(batch) >= B:
                out.append(np.array(list(model.embed(batch, batch_size=8)), dtype=np.float16)); batch=[]
                if time.time()-last_save > CHECKPOINT_S: flush()
                if time.time()-t0 > limit_s: break
        if batch and time.time()-t0 <= limit_s:
            out.append(np.array(list(model.embed(batch, batch_size=8)), dtype=np.float16))
    flush()
    return saved >= total

W_,OV_=1800,200
def make_windows():
    if os.path.exists(WFILE) and os.path.getsize(WFILE)>1000: return
    print('creating window plan...', flush=True)
    n=0
    with open(CORPUS, encoding='utf-8') as f, open(WFILE,'w',encoding='utf-8') as wf:
        for ri,line in enumerate(f):
            r=json.loads(line); t=r['text']; pos=0
            while pos<len(t):
                end=min(len(t),pos+W_)
                wf.write(json.dumps({'ri':ri,'ref':r['ref'],'a':pos,'b':end},ensure_ascii=False)+'\n'); n+=1
                if end==len(t): break
                pos=end-OV_
    print('windows:',n, flush=True)

if __name__ == '__main__':
    print('build_index starting (python OK)...', flush=True)
    os.makedirs(IDX, exist_ok=True)
    make_windows()
    make_texts()
    print('window texts present', flush=True)
    full = embed(float(sys.argv[1]) if len(sys.argv)>1 else 30)
    print('COMPLETE' if full else 'PARTIAL')
    sys.exit(0 if full else 3)

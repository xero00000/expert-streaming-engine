#!/usr/bin/env python3
"""Fetch only the pinned original V4.1 DSpark tensors using bounded ranges."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import struct
import time
from deepseek_v41_preflight import read_range
from deepseek_v41_dense_repair import atomic
from deepseek_v41_package import sha

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--reference',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
provenance=json.loads((a.reference/'provenance.json').read_text())
documents={};selected={}
for path in (a.reference/'headers').glob('*.json'):
 d=json.loads(path.read_text());assert d['revision']==provenance['revision']
 for name,item in d['header'].items():
  if name.startswith('mtp.'):
   selected[name]={'shard':d['shard'],**item};documents[d['shard']]=d
assert selected and {name.split('.')[1] for name in selected}=={'0','1','2'}
mp=a.output/'manifest.json'
m=json.loads(mp.read_text()) if mp.exists() else {'repository':provenance['repository'],
 'revision':provenance['revision'],'ranges':{},'tensors':selected}
assert m['revision']==provenance['revision'] and m['tensors']==selected
m.update(status='fetching',pid=os.getpid());atomic(mp,m)
def fetch(shard,start,size):
 assert 0<size<=64<<20
 key=f'{shard}.{start}.{size}.bin';path=a.output/key
 old=m['ranges'].get(key)
 if old and path.exists():
  assert path.stat().st_size==size and sha(path)==old['sha256'];return key,old
 url=f"https://huggingface.co/{m['repository']}/resolve/{m['revision']}/{shard}?ese_dspark={start}-{size}"
 for attempt in range(5):
  try:data=read_range(url,start,size);break
  except Exception:
   if attempt==4:raise
   time.sleep(2**attempt)
 tmp=path.with_suffix('.partial');tmp.write_bytes(data);tmp.replace(path)
 return key,{'shard':shard,'offset':start,'bytes':size,'sha256':hashlib.sha256(data).hexdigest()}
bases={}
for shard,d in documents.items():
 key,r=fetch(shard,0,8);m['ranges'][key]=r
 size=struct.unpack('<Q',(a.output/key).read_bytes())[0];assert size<32<<20
 key,r=fetch(shard,8,size);assert r['sha256']==d['header_sha256'];m['ranges'][key]=r
 bases[shard]=size+8;atomic(mp,m)
m['data_bases']=bases
chunks=[]
for shard in documents:
 groups=[]
 for start,end in sorted(v['data_offsets'] for v in selected.values() if v['shard']==shard):
  if groups and groups[-1][1]==start:groups[-1][1]=end
  else:groups.append([start,end])
 for start,end in groups:
  while start<end:
   size=min(64<<20,end-start);chunks.append((shard,bases[shard]+start,size));start+=size
m['payload_bytes']=sum(c[2] for c in chunks);atomic(mp,m)
print(f'Fetching {len(selected)} DSpark tensors, {m["payload_bytes"]:,} bytes in {len(chunks)} ranges',flush=True)
with ThreadPoolExecutor(max_workers=3) as pool:
 for index,future in enumerate(as_completed([pool.submit(fetch,*c) for c in chunks]),1):
  key,r=future.result();m['ranges'][key]=r;atomic(mp,m)
  print(f'Verified {index}/{len(chunks)}',flush=True)
m.update(status='complete',finished=time.time());atomic(mp,m)
print('Pinned DSpark ranges complete',flush=True)

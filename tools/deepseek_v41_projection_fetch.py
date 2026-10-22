#!/usr/bin/env python3
"""Fetch only original Engram projection/norm ranges from pinned model shards."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import time
from deepseek_v41_preflight import read_range
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('snapshot',type=Path);p.add_argument('output',type=Path)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
provenance=json.loads((a.snapshot/'provenance.json').read_text())
records=[]
for layer, shard in [(1,47),(14,48)]:
 name=f'model-{shard:05d}-of-00048.safetensors'
 expected=json.loads((a.snapshot/'headers'/(name+'.json')).read_text())
 base=f"https://huggingface.co/{provenance['repository']}/resolve/{provenance['revision']}/{name}"
 def fetch(start,size):
  for attempt in range(5):
   try:return read_range(base+f'?ese_projection={start}-{size}',start,size)
   except Exception:
    if attempt==4:raise
    time.sleep(2**attempt)
 header_size=struct.unpack('<Q',fetch(0,8))[0]
 raw=fetch(8,header_size)
 assert hashlib.sha256(raw).hexdigest()==expected['header_sha256']
 header=json.loads(raw)
 for suffix in ['q_weight','k_weight','wkv.weight','wkv.scale']:
  tensor=f'layers.{layer}.engram.{suffix}'
  item=header[tensor];start,end=item['data_offsets'];size=end-start
  final=a.output/(tensor+'.bin');partial=final.with_suffix('.partial')
  if not final.exists():
   have=partial.stat().st_size if partial.exists() else 0
   assert have<=size
   with partial.open('ab') as stream:
    while have<size:
     chunk=fetch(8+header_size+start+have,min(8<<20,size-have));stream.write(chunk);stream.flush();have+=len(chunk)
     print(f'{tensor}: {have}/{size}',flush=True)
   partial.replace(final)
  assert final.stat().st_size==size
  digest=hashlib.file_digest(final.open('rb'),'sha256').hexdigest()
  records.append(dict(name=tensor,file=final.name,bytes=size,sha256=digest,dtype=item['dtype'],shape=item['shape'],source_shard=name,source_offset=8+header_size+start))
(a.output/'manifest.json').write_text(json.dumps(dict(repository=provenance['repository'],revision=provenance['revision'],records=records),indent=2)+'\n')
print('Projection ranges complete',flush=True)

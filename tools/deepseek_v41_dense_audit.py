#!/usr/bin/env python3
"""Compare bounded trained dense-weight samples against pinned original ranges."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import struct
import time
import numpy as np
from deepseek_v41_package import header_parts, expand_fp8
from deepseek_v41_preflight import read_range

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--reference',type=Path,required=True)
p.add_argument('--package',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
provenance=json.loads((a.reference/'provenance.json').read_text())
source={};documents={};native={}
for f in (a.reference/'headers').glob('*.json'):
 d=json.loads(f.read_text());documents[d['shard']]=d
 for name,item in d['header'].items():
  if name!='__metadata__':source[name]=(d['shard'],item)
for f in a.package.glob('*.gguf'):
 h=header_parts(f)
 for t in h[5]:native[t['name']]=(f,h[3]+t['relative_offset'],t)
pairs=[('embed.weight','token_embd'),('head.weight','output'),('norm.weight','output_norm')]
for src,dst in [('attn.wq_a.weight','attn_q_a'),('attn.wq_b.weight','attn_q_b'),
 ('attn.wkv.weight','attn_kv'),('attn.wo_a.weight','attn_output_a'),('attn.wo_b.weight','attn_output_b'),
 ('attn_norm.weight','attn_norm'),('ffn.gate.weight','ffn_gate_inp'),('ffn.gate.bias','exp_probs_b'),
 ('hc_attn_fn','hc_attn_fn'),('hc_attn_scale','hc_attn_scale'),('hc_attn_base','hc_attn_base')]:
 pairs.append(('layers.0.'+src,'blk.0.'+dst))
def fetch(shard,start,size):
 key=f'{shard}.{start}.{size}.bin';dest=a.output/key
 if dest.exists():
  data=dest.read_bytes();assert len(data)==size;return data
 url=f"https://huggingface.co/{provenance['repository']}/resolve/{provenance['revision']}/{shard}?ese_dense_audit={start}-{size}"
 for attempt in range(5):
  try:data=read_range(url,start,size);break
  except Exception:
   if attempt==4:raise
   time.sleep(2**attempt)
 dest.write_bytes(data);return data
bases={}
for shard in sorted({source[name][0] for name,_ in pairs}):
 size=struct.unpack('<Q',fetch(shard,0,8))[0]
 assert size<32<<20
 raw=fetch(shard,8,size)
 assert hashlib.sha256(raw).hexdigest()==documents[shard]['header_sha256']
 bases[shard]=size+8
def compare(pair):
 name,target=pair;shard,item=source[name];path,offset,tensor=native[target]
 assert list(reversed(tensor['shape']))==item['shape'],(name,'shape')
 shape=item['shape'];rows=shape[0] if len(shape)>1 else 1;width=shape[-1]
 starts=sorted({0,((rows//2)//32)*32,((rows-1)//32)*32}) if len(shape)>1 else [0]
 records=[]
 for row in starts:
  count=min(32,rows-row);dtype=item['dtype'];element=1 if dtype=='F8_E4M3' else 2 if dtype=='BF16' else 4
  raw=fetch(shard,bases[shard]+item['data_offsets'][0]+row*width*element,count*width*element)
  if dtype=='F8_E4M3':
   s_shard,scale=source[name.replace('.weight','.scale')];assert s_shard==shard
   scale_width=scale['shape'][1]
   sr=fetch(shard,bases[shard]+scale['data_offsets'][0]+(row//32)*scale_width,scale_width)
   expected=expand_fp8(np.frombuffer(raw,np.uint8).reshape(count,width),np.frombuffer(sr,np.uint8).reshape(1,scale_width)).tobytes()
  else:expected=raw
  with path.open('rb') as stream:
   stream.seek(offset+row*width*(4 if dtype=='F32' else 2));actual=stream.read(len(expected))
  records.append({'row':row,'rows':count,'bytes':len(expected),'equal':actual==expected,
   'source_sha256':hashlib.sha256(raw).hexdigest(),'expected_sha256':hashlib.sha256(expected).hexdigest(),
   'native_sha256':hashlib.sha256(actual).hexdigest()})
 result={'source':name,'native':target,'dtype':dtype,'samples':records,'equal':all(r['equal'] for r in records)}
 print(name,result['equal'],flush=True);return result
with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(compare,pairs))
report={'repository':provenance['repository'],'revision':provenance['revision'],
 'scope':'bounded dense samples; FP8 expanded with original 32x32 scales; routed experts excluded',
 'results':results,'equal':all(r['equal'] for r in results)}
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
raise SystemExit(0 if report['equal'] else 1)

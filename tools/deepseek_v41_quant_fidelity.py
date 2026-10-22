#!/usr/bin/env python3
"""Measure layer-0 expert-0 quantization error against original MXFP4 rows."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import time
import numpy as np
from deepseek_v41_package import header_parts, gguf
from deepseek_v41_preflight import read_range

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--reference',type=Path,required=True)
p.add_argument('--package',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
provenance=json.loads((a.reference/'provenance.json').read_text())
shard='model-00003-of-00048.safetensors'
d=json.loads((a.reference/'headers'/(shard+'.json')).read_text());h=d['header']
def fetch(start,size):
 assert 0<size<=1<<20
 dest=a.output/f'{start}.{size}.bin'
 if dest.exists():data=dest.read_bytes();assert len(data)==size;return data
 url=f"https://huggingface.co/{provenance['repository']}/resolve/{provenance['revision']}/{shard}?ese_quant_fidelity={start}-{size}"
 for attempt in range(5):
  try:data=read_range(url,start,size);break
  except Exception:
   if attempt==4:raise
   time.sleep(2**attempt)
 dest.write_bytes(data);return data
size=struct.unpack('<Q',fetch(0,8))[0]
assert hashlib.sha256(fetch(8,size)).hexdigest()==d['header_sha256']
base=size+8
path=a.package/'DeepSeek-V4.1-Flash-MixedQ2-00001-of-00006.gguf'
nh=header_parts(path);tensors={t['name']:t for t in nh[5]}
codebook=np.array([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6],np.float32)
results=[]
for stem,target in [('w1','ffn_gate_exps'),('w3','ffn_up_exps'),('w2','ffn_down_exps')]:
 name='layers.0.ffn.experts.0.'+stem;w=h[name+'.weight'];s=h[name+'.scale']
 nrows,packed_width=w['shape'];width=packed_width*2
 assert w['dtype']=='I8' and s['dtype']=='F8_E8M0' and s['shape']==[nrows,width//32]
 t=tensors['blk.0.'+target];kind=gguf.GGMLQuantizationType(t['type_id'])
 block,block_bytes=gguf.GGML_QUANT_SIZES[kind];row_bytes=width//block*block_bytes
 assert t['shape'][:2]==[width,nrows]
 for row in [0,nrows//2,nrows-32]:
  raw=fetch(base+w['data_offsets'][0]+row*packed_width,32*packed_width)
  scales=fetch(base+s['data_offsets'][0]+row*(width//32),width)
  packed=np.frombuffer(raw,np.uint8).reshape(32,packed_width)
  codes=np.empty((32,width),np.uint8);codes[:,0::2]=packed&15;codes[:,1::2]=packed>>4
  scale=np.exp2(np.frombuffer(scales,np.uint8).astype(np.float32)-127).reshape(32,-1).repeat(32,axis=1)
  original=codebook[codes]*scale
  with path.open('rb') as stream:
   stream.seek(nh[3]+t['relative_offset']+row*row_bytes);quant=stream.read(32*row_bytes)
  actual=gguf.dequantize(np.frombuffer(quant,np.uint8).reshape(32,row_bytes),kind)
  cosine=float(np.sum(original*actual)/np.sqrt(np.sum(original**2)*np.sum(actual**2)))
  error=float(np.linalg.norm(actual-original)/np.linalg.norm(original))
  r={'source':name,'native':'blk.0.'+target,'row':row,'rows':32,'type':kind.name,
     'cosine':cosine,'relative_rms':error,'source_sha256':hashlib.sha256(raw).hexdigest(),
     'scales_sha256':hashlib.sha256(scales).hexdigest(),'native_sha256':hashlib.sha256(quant).hexdigest()}
  results.append(r);print(target,row,cosine,error,flush=True)
(a.output/'report.json').write_text(json.dumps({'repository':provenance['repository'],
 'revision':provenance['revision'],'scope':'layer 0 expert 0 only; three 32-row samples per projection',
 'results':results},indent=2)+'\n')

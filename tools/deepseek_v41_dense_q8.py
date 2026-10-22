#!/usr/bin/env python3
"""Create a reflinked Q8_0-dense variant, preserving routed experts and Engram.

Only the 328 repaired FP8-derived dense matrices are quantized. Tensor offsets
remain fixed; unused trailing bytes are ignored by GGUF readers. The source
package is never opened for writing. Requires copy-on-write reflink support.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import numpy as np
from deepseek_v41_package import header_parts, gguf, sha
from deepseek_v41_gguf_probe import PrefixReader
from deepseek_v41_dense_repair import atomic, file_hashes


def type_positions(path):
    header=header_parts(path);r=PrefixReader(header[6])
    assert r.take(4)==b'GGUF' and r.scalar(4)==3
    nt,nm=r.scalar(10),r.scalar(10)
    for _ in range(nm):r.string();r.value(r.scalar(4),False)
    result={}
    for _ in range(nt):
        name=r.string();nd=r.scalar(4)
        for _ in range(nd):r.scalar(10)
        result[name]=r.offset;r.scalar(4);r.scalar(10)
    return header,result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--repaired-tensors',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert a.source.resolve()!=a.output.resolve()
    source=json.loads((a.source/'package.json').read_text());assert source['status']=='complete'
    prepared=json.loads(a.repaired_tensors.read_text())
    assert source['dense_repair']['prepared_sha256']==sha(a.repaired_tensors)
    assert len(prepared['records'])==328
    a.output.mkdir(parents=True,exist_ok=True);mp=a.output/'package.json'
    source_sha=sha(a.source/'package.json')
    if mp.exists():
        m=json.loads(mp.read_text());assert m['variant_source_manifest_sha256']==source_sha
    else:
        m=copy.deepcopy(source);m.update(status='building_dense_q8',variant_source=str(a.source.resolve()),
            variant_source_manifest_sha256=source_sha,dense_quantization={'type':'Q8_0','records':[]})
        atomic(mp,m)
    for entry in source['files']:
        dest=a.output/entry['file']
        if not dest.exists():
            tmp=dest.with_suffix(dest.suffix+'.clone-partial')
            subprocess.run(['cp','--reflink=always','--',str(a.source/entry['file']),str(tmp)],check=True)
            tmp.replace(dest)
        assert dest.stat().st_size==entry['bytes']
    headers={};positions={}
    for f in a.source.glob('*.gguf'):headers[f.name],positions[f.name]=type_positions(f)
    completed={r['native']:r for r in m['dense_quantization']['records']}
    for item in prepared['records']:
        name=item['native'];target=a.output/item['file']
        if name in completed:
            record=completed[name]
            with target.open('rb') as f:
                f.seek(item['offset']);assert hashlib.sha256(f.read(record['bytes'])).hexdigest()==record['sha256']
                f.seek(positions[item['file']][name]);assert struct.unpack('<I',f.read(4))[0]==8
            continue
        h=headers[item['file']];t=next(t for t in h[5] if t['name']==name)
        assert t['type_id']==30 and len(t['shape'])==2
        width,rows=t['shape'];assert width%32==0
        original_hash=hashlib.sha256();quant_hash=hashlib.sha256()
        sum_original=sum_actual=sum_product=sum_error=0.0;written=0
        with (a.source/item['file']).open('rb') as src,target.open('r+b') as dst:
            src.seek(item['offset']);dst.seek(item['offset'])
            for row in range(0,rows,256):
                count=min(256,rows-row);raw=src.read(count*width*2);assert len(raw)==count*width*2
                original_hash.update(raw)
                values=(np.frombuffer(raw,'<u2').astype(np.uint32)<<16).view(np.float32).reshape(count,width)
                quant=gguf.quantize(values,gguf.GGMLQuantizationType.Q8_0)
                actual=gguf.dequantize(quant,gguf.GGMLQuantizationType.Q8_0)
                sum_original+=float(np.sum(values*values,dtype=np.float64))
                sum_actual+=float(np.sum(actual*actual,dtype=np.float64))
                sum_product+=float(np.sum(actual*values,dtype=np.float64))
                sum_error+=float(np.sum((actual-values)**2,dtype=np.float64))
                payload=quant.tobytes();dst.write(payload);quant_hash.update(payload);written+=len(payload)
            assert original_hash.hexdigest()==item['sha256'],'original dense weight changed'
            assert written==rows*(width//32)*34 and written<item['bytes']
            dst.flush();os.fsync(dst.fileno())
            dst.seek(item['offset']);assert hashlib.sha256(dst.read(written)).hexdigest()==quant_hash.hexdigest()
            dst.seek(positions[item['file']][name]);dst.write(struct.pack('<I',8));dst.flush();os.fsync(dst.fileno())
        record={'native':name,'file':item['file'],'offset':item['offset'],'bytes':written,
            'source_bf16_sha256':original_hash.hexdigest(),'sha256':quant_hash.hexdigest(),
            'relative_rms':float(np.sqrt(sum_error/sum_original)),
            'cosine':float(sum_product/np.sqrt(sum_original*sum_actual))}
        m['dense_quantization']['records'].append(record);atomic(mp,m)
        print(f'{len(m["dense_quantization"]["records"])}/328 {name} RMS={record["relative_rms"]:.6f}',flush=True)
    affected={r['file'] for r in prepared['records']}
    for entry in m['files']:
        if entry['file'] not in affected:continue
        m['current']='hashing '+entry['file'];atomic(mp,m)
        digest,payload=file_hashes(a.output/entry['file'])
        old=next(r for r in source['files'] if r['file']==entry['file'])
        entry.update(pre_dense_q8_sha256=old['sha256'],sha256=digest,tensor_payload_sha256=payload)
        print('Verified',entry['file'],flush=True)
    m.update(status='complete');m.pop('current',None);atomic(mp,m)
    print('Q8 dense variant complete',flush=True)


if __name__=='__main__':main()

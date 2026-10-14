#!/usr/bin/env python3
"""Fetch and expand original dense FP8 tensors; optionally repair native GGUFs.

Original quant downloads remain untouched. The native package is unavailable
while patching, and completed tensor writes are journaled for safe replay.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import time
import numpy as np
from deepseek_v41_package import header_parts, expand_fp8, sha, tensor_bytes
from deepseek_v41_preflight import read_range


def atomic(path, value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def patch_tensor(path, offset, payload):
    """Write and reread only the declared tensor extent; replay is idempotent."""
    assert offset>=0 and offset+len(payload)<=path.stat().st_size
    with path.open('r+b') as f:
        f.seek(offset);assert f.write(payload)==len(payload);f.flush();os.fsync(f.fileno())
    with path.open('rb') as f:
        f.seek(offset);actual=f.read(len(payload))
    assert actual==payload,'dense tensor write verification failed'


def file_hashes(path):
    h=header_parts(path);all_hash=hashlib.sha256();payload=hashlib.sha256()
    with path.open('rb') as f:
        all_hash.update(f.read(h[3]));position=0
        for t in sorted(h[5],key=lambda t:t['relative_offset']):
            gap=t['relative_offset']-position;assert gap>=0
            all_hash.update(f.read(gap))
            remaining=tensor_bytes(t);position=t['relative_offset']+remaining
            while remaining:
                n=min(remaining,32<<20);chunk=f.read(n);assert len(chunk)==n
                all_hash.update(chunk);payload.update(chunk);remaining-=n
        while chunk:=f.read(32<<20):all_hash.update(chunk)
    return all_hash.hexdigest(),payload.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--package',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--apply',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    provenance=json.loads((a.reference/'provenance.json').read_text())
    assert re.fullmatch('[a-f0-9]{40}',provenance['revision'])
    source={};documents={};native={}
    for f in (a.reference/'headers').glob('*.json'):
        d=json.loads(f.read_text());assert d['revision']==provenance['revision']
        documents[d['shard']]=d
        for name,item in d['header'].items():
            if name!='__metadata__':source[name]=(d['shard'],item)
    for f in a.package.glob('*.gguf'):
        h=header_parts(f)
        for t in h[5]:native[t['name']]=(f,h[3]+t['relative_offset'],t)
    names={'attn.wq_a':'attn_q_a','attn.wq_b':'attn_q_b','attn.wkv':'attn_kv',
           'attn.wo_a':'attn_output_a','attn.wo_b':'attn_output_b',
           'attn.indexer.wq_b':'indexer.attn_q_b','ffn.shared_experts.w1':'ffn_gate_shexp',
           'ffn.shared_experts.w2':'ffn_down_shexp','ffn.shared_experts.w3':'ffn_up_shexp'}
    pairs=[];selected=set()
    for name,(shard,item) in source.items():
        if not name.startswith('layers.') or '.engram.' in name or item['dtype']!='F8_E4M3':continue
        match=re.fullmatch(r'layers\.(\d+)\.(.+)\.weight',name);assert match
        layer,stem=match.groups();assert stem in names,(name,'unmapped FP8 tensor')
        target=f'blk.{layer}.{names[stem]}';path,offset,t=native[target]
        assert t['type_id']==30 and list(reversed(t['shape']))==item['shape'],(name,'native shape/type')
        scale=name[:-6]+'scale';scale_shard,s=source[scale]
        assert scale_shard==shard and s['dtype']=='F8_E8M0'
        assert s['shape']==[item['shape'][0]//32,item['shape'][1]//32]
        pairs.append((name,scale,target));selected.update((name,scale))
    assert len(pairs)==328,'unexpected dense FP8 inventory'
    shards=sorted({source[name][0] for name in selected})
    bases={};download_path=a.output/'downloads.json'
    downloads=json.loads(download_path.read_text()) if download_path.exists() else {}

    def fetch(shard,start,size):
        assert 0<size<=64<<20
        key=f'{shard}.{start}.{size}.bin';dest=a.output/key
        if key in downloads and dest.exists():
            assert dest.stat().st_size==size and sha(dest)==downloads[key]['sha256']
            return key,downloads[key]
        url=f"https://huggingface.co/{provenance['repository']}/resolve/{provenance['revision']}/{shard}?ese_dense_repair={start}-{size}"
        for attempt in range(5):
            try:data=read_range(url,start,size);break
            except Exception:
                if attempt==4:raise
                time.sleep(2**attempt)
        tmp=dest.with_suffix('.partial');tmp.write_bytes(data);tmp.replace(dest)
        return key,{'sha256':hashlib.sha256(data).hexdigest(),'bytes':size,'shard':shard,'offset':start}

    def header(shard):
        key,r1=fetch(shard,0,8);size=struct.unpack('<Q',(a.output/key).read_bytes())[0]
        assert size<32<<20
        key2,r2=fetch(shard,8,size)
        assert r2['sha256']==documents[shard]['header_sha256']
        return shard,size+8,{key:r1,key2:r2}

    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(header,s) for s in shards]):
            shard,base,records=future.result();bases[shard]=base;downloads.update(records);atomic(download_path,downloads)
    chunks=[]
    for shard in shards:
        intervals=sorted(source[name][1]['data_offsets'] for name in selected if source[name][0]==shard)
        groups=[]
        for start,end in intervals:
            if groups and groups[-1][1]==start:groups[-1][1]=end
            else:groups.append([start,end])
        for start,end in groups:
            while start<end:
                n=min(64<<20,end-start);chunks.append((shard,bases[shard]+start,n));start+=n
    print(f'Fetching {sum(n for _,_,n in chunks):,} source bytes in {len(chunks)} bounded ranges',flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        completed=0
        for future in as_completed([pool.submit(fetch,*chunk) for chunk in chunks]):
            key,record=future.result();downloads[key]=record;atomic(download_path,downloads);completed+=1
            print(f'Verified range {completed}/{len(chunks)}',flush=True)

    def tensor(name):
        shard,item=source[name];start,end=item['data_offsets'];start+=bases[shard];end+=bases[shard]
        result=bytearray()
        for s,begin,size in chunks:
            if s!=shard:continue
            left=max(start,begin);right=min(end,begin+size)
            if left>=right:continue
            with (a.output/f'{s}.{begin}.{size}.bin').open('rb') as f:
                f.seek(left-begin);result.extend(f.read(right-left))
        assert len(result)==end-start
        return np.frombuffer(result,np.uint8).reshape(item['shape'])

    records=[]
    for name,scale,target in sorted(pairs):
        raw=tensor(name);scales=tensor(scale);expanded=expand_fp8(raw,scales).tobytes()
        dest=a.output/(target+'.bf16');digest=hashlib.sha256(expanded).hexdigest()
        if dest.exists():assert dest.stat().st_size==len(expanded) and sha(dest)==digest
        else:
            tmp=dest.with_suffix('.partial');tmp.write_bytes(expanded);tmp.replace(dest)
        path,offset,t=native[target];assert len(expanded)==tensor_bytes(t)
        records.append({'source':name,'source_sha256':hashlib.sha256(raw).hexdigest(),
            'scale_sha256':hashlib.sha256(scales).hexdigest(),'native':target,'file':path.name,
            'offset':offset,'bytes':len(expanded),'payload':dest.name,'sha256':digest})
    prepared={'repository':provenance['repository'],'revision':provenance['revision'],'records':records}
    atomic(a.output/'prepared.json',prepared)
    print(f'Prepared {len(records)} original dense tensors, {sum(r["bytes"] for r in records):,} bytes',flush=True)
    if not a.apply:return

    manifest_path=a.package/'package.json';manifest=json.loads(manifest_path.read_text())
    journal_path=a.package/'dense-repair.json'
    if journal_path.exists():
        journal=json.loads(journal_path.read_text())
        assert journal['prepared_sha256']==sha(a.output/'prepared.json')
    else:
        assert manifest['status']=='complete'
        journal={'prepared_sha256':sha(a.output/'prepared.json'),'completed':[],
                 'original_files':manifest['files'],'started':time.time()}
        atomic(journal_path,journal)
    manifest.update(status='repairing_dense',pid=os.getpid());atomic(manifest_path,manifest)
    completed=set(journal['completed'])
    for r in records:
        path=a.package/r['file'];payload=(a.output/r['payload']).read_bytes()
        assert hashlib.sha256(payload).hexdigest()==r['sha256']
        if r['native'] in completed:
            with path.open('rb') as f:f.seek(r['offset']);assert f.read(r['bytes'])==payload
            continue
        manifest['current']=r['native'];atomic(manifest_path,manifest)
        patch_tensor(path,r['offset'],payload)
        journal['completed'].append(r['native']);atomic(journal_path,journal)
        print('Replaced and verified',r['native'],flush=True)
    affected={r['file'] for r in records}
    original={r['file']:r for r in journal['original_files']}
    for r in manifest['files']:
        if r['file'] not in affected:continue
        manifest['current']='hashing '+r['file'];atomic(manifest_path,manifest)
        digest,payload=file_hashes(a.package/r['file'])
        r.update(pre_dense_repair_sha256=original[r['file']]['sha256'],sha256=digest,
                 pre_dense_repair_tensor_payload_sha256=original[r['file']]['tensor_payload_sha256'],
                 tensor_payload_sha256=payload,dense_repair_tensors=sum(x['file']==r['file'] for x in records))
        print('Rehashed',r['file'],flush=True)
    manifest.update(status='complete',dense_repair={'repository':provenance['repository'],
        'revision':provenance['revision'],'prepared_sha256':sha(a.output/'prepared.json'),
        'tensors':len(records),'finished':time.time()})
    manifest.pop('current',None);atomic(manifest_path,manifest)
    print('Dense repair complete',flush=True)


if __name__=='__main__':main()

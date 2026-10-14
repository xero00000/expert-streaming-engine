#!/usr/bin/env python3
"""Stage canonical native V4.1 GGUF shards and byte-preserving affine2 row files.

Backbone tensor payloads remain byte-identical; padding is realigned. Original FP8 Engram projections
are expanded to BF16; large affine2 tables retain their exact quantization.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import time
import numpy as np
from deepseek_v41_gguf_probe import PrefixReader, parse_header
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gguf-py'))
import gguf


def string(value):
    raw=value.encode(); return struct.pack('<Q',len(raw))+raw

def record(key,kind,value):
    encoded=string(value) if kind==8 else struct.pack('<'+{2:'H',4:'I',5:'i',6:'f',7:'?',10:'Q'}[kind],value)
    return string(key)+struct.pack('<I',kind)+encoded

def header_parts(path):
    with Path(path).open('rb') as f: data=f.read(16<<20)
    r=PrefixReader(data)
    assert r.take(4)==b'GGUF' and r.scalar(4)==3
    count,nmeta=r.scalar(10),r.scalar(10)
    records={}
    for _ in range(nmeta):
        start=r.offset;key=r.string();kind=r.scalar(4);r.value(kind)
        assert key not in records
        records[key]=data[start:r.offset]
    tensors_start=r.offset
    for _ in range(count):
        r.string();nd=r.scalar(4)
        for _ in range(nd):r.scalar(10)
        r.scalar(4);r.scalar(10)
    meta,tensors,offset=parse_header(data)
    return records,data[tensors_start:r.offset],count,offset,meta,tensors,data[:offset]

def tensor_bytes(tensor):
    block,size=gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(tensor['type_id'])]
    count=int(np.prod(tensor['shape'],dtype=object))
    assert tensor['shape'][0]%block==0
    return count//block*size

def canonical_header(path,extra):
    records,_,count,offset,meta,tensors,raw=header_parts(path)
    records={k:v for k,v in records.items() if not k.startswith('deepseek4.')}
    records.update(extra)
    alignment=meta.get('general.alignment',32)
    cursor=0;offsets={}
    for t in sorted(tensors,key=lambda x:x['relative_offset']):
        cursor+=(-cursor)%alignment;offsets[t['name']]=cursor;cursor+=tensor_bytes(t)
    tensor_info=b''
    for t in tensors:
        tensor_info+=string(t['name'])+struct.pack('<I',len(t['shape']))
        tensor_info+=b''.join(struct.pack('<Q',d) for d in t['shape'])
        tensor_info+=struct.pack('<IQ',t['type_id'],offsets[t['name']])
    header=b'GGUF'+struct.pack('<IQQ',3,count,len(records))+b''.join(records.values())+tensor_info
    header+=b'\0'*((-len(header))%alignment)
    return header,offset,raw,tensors

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def copy_backbone(source,destination,extra,expected_sha):
    header,offset,raw,tensors=canonical_header(source,extra)
    _,rewritten,new_base=parse_header(header)
    new_by_name={t['name']:t for t in rewritten}
    ordered=sorted(tensors,key=lambda t:t['relative_offset'])
    partial=destination.with_suffix('.partial')
    covered=0
    if partial.exists():
        with partial.open('rb') as f:assert f.read(len(header))==header,'incompatible partial header'
        covered=partial.stat().st_size-len(header)
    else:partial.write_bytes(header)
    source_hash=hashlib.sha256(raw);payload_hash=hashlib.sha256()
    with source.open('rb') as src,partial.open('r+b') as dst:
        src.seek(offset);dst.seek(len(header));source_pos=0;dest_pos=0;progress=0
        def source_chunk(n,payload=False):
            chunk=src.read(n);assert len(chunk)==n,'truncated source tensor'
            source_hash.update(chunk)
            if payload:payload_hash.update(chunk)
            return chunk
        def emit(chunk):
            nonlocal dest_pos
            old=min(len(chunk),max(0,covered-dest_pos))
            if old:assert dst.read(old)==chunk[:old],'partial tensor data mismatch'
            if old<len(chunk):dst.write(chunk[old:])
            dest_pos+=len(chunk)
        for t in ordered:
            gap=t['relative_offset']-source_pos;assert gap>=0,'overlapping source tensors'
            while gap:
                n=min(gap,32<<20);source_chunk(n);gap-=n;source_pos+=n
            target=new_by_name[t['name']]['relative_offset']
            emit(b'\0'*(target-dest_pos))
            remaining=tensor_bytes(t)
            while remaining:
                n=min(remaining,32<<20);emit(source_chunk(n,True));remaining-=n;source_pos+=n
                if dest_pos//(1<<30)>progress:
                    progress=dest_pos//(1<<30);print(f'{destination.name}: {dest_pos/1e9:.1f} GB copied',flush=True)
        while chunk:=src.read(32<<20):source_hash.update(chunk)
        dst.truncate(len(header)+dest_pos);dst.flush();os.fsync(dst.fileno())
    assert source_hash.hexdigest()==expected_sha,'source file checksum changed'
    verify=hashlib.sha256();all_hash=hashlib.sha256()
    with partial.open('rb') as f:
        actual_header=f.read(len(header));assert actual_header==header;all_hash.update(actual_header);position=0
        for t in ordered:
            target=new_by_name[t['name']]['relative_offset']
            gap=f.read(target-position);assert len(gap)==target-position and not any(gap);all_hash.update(gap)
            remaining=tensor_bytes(t);position=target+remaining
            while remaining:
                n=min(remaining,32<<20);chunk=f.read(n);assert len(chunk)==n
                verify.update(chunk);all_hash.update(chunk);remaining-=n
        assert f.read(1)==b''
    assert verify.digest()==payload_hash.digest(),'destination tensor payload differs'
    partial.replace(destination)
    return dict(file=destination.name,bytes=destination.stat().st_size,sha256=all_hash.hexdigest(),
                source_sha256=expected_sha,tensor_payload_sha256=payload_hash.hexdigest(),tensors=len(tensors),alignment=32)

def safetensor_header(path):
    with Path(path).open('rb') as f:
        size=struct.unpack('<Q',f.read(8))[0];assert size<32<<20
        return json.loads(f.read(size)),size+8

def pack_rows(fd,base,header,stem,start,count):
    output=np.empty((count,80),np.uint8)
    for suffix,width,destination in [('weight',64,0),('scales',8,64),('biases',8,72)]:
        entry=header[stem+'.'+suffix]
        data=os.pread(fd,count*width,base+entry['data_offsets'][0]+start*width)
        assert len(data)==count*width,'truncated row source'
        output[:,destination:destination+width]=np.frombuffer(data,np.uint8).reshape(count,width)
    return output

def convert_rows(source,destination,layer,rows):
    header,base=safetensor_header(source);stem=f'layers.{layer}.engram.embed'
    for suffix,width,dtype in [('weight',16,'U32'),('scales',4,'BF16'),('biases',4,'BF16')]:
        assert header[stem+'.'+suffix]['shape']==[rows,width] and header[stem+'.'+suffix]['dtype']==dtype
    partial=destination.with_suffix('.partial')
    done=partial.stat().st_size//80 if partial.exists() else 0
    assert not partial.exists() or partial.stat().st_size==done*80
    assert done<=rows
    with source.open('rb') as src,partial.open('ab') as dst:
        if done:
            n=min(done,4096)
            with partial.open('rb') as old:
                old.seek((done-n)*80);assert old.read(n*80)==pack_rows(src.fileno(),base,header,stem,done-n,n).tobytes()
        while done<rows:
            count=min(1<<20,rows-done)
            data=pack_rows(src.fileno(),base,header,stem,done,count)
            dst.write(data.tobytes());done+=count
            if done%(16<<20)==0:print(f'{destination.name}: {done}/{rows} rows',flush=True)
        dst.flush();os.fsync(dst.fileno())
    assert partial.stat().st_size==rows*80
    # Check byte-exact original planes at beginning, end and deterministic interior rows.
    with source.open('rb') as src,partial.open('rb') as dst:
        for row in sorted({0,1,rows-1,*[(i*104729)%rows for i in range(128)]}):
            dst.seek(row*80);assert dst.read(80)==pack_rows(src.fileno(),base,header,stem,row,1).tobytes()
    digest=sha(partial);partial.replace(destination)
    return dict(file=destination.name,bytes=rows*80,sha256=digest,rows=rows,encoding='affine2-g64-bf16')

def expand_fp8(weights,scales):
    code=np.arange(256,dtype=np.int32);exp=(code>>3)&15;mant=code&7
    lut=np.where(exp==0,mant*2.0**-9,(1+mant/8.0)*np.exp2(exp-7)).astype(np.float32)
    lut=np.where(code&128,-lut,lut);lut[(exp==15)&(mant==7)]=np.nan
    out=np.empty(weights.shape,np.uint16)
    for start in range(0,weights.shape[0],256):
        end=min(start+256,weights.shape[0])
        rows=np.arange(start,end)//32
        factor=np.exp2(scales[rows].astype(np.float32)-127).repeat(32,axis=1)
        values=lut[weights[start:end]]*factor
        assert np.isfinite(values).all()
        bits=values.astype('<f4').view('<u4');out[start:end]=((bits+0x7fff+((bits>>16)&1))>>16).astype('<u2')
    return out

def projection_shard(directory,destination,split_count,total):
    manifest=json.loads((directory/'manifest.json').read_text())
    items={r['name']:r for r in manifest['records']}
    writer=gguf.GGUFWriter(destination.with_suffix('.partial'),'deepseek41')
    writer.add_uint16('split.no',split_count-1);writer.add_uint16('split.count',split_count);writer.add_int32('split.tensors.count',total)
    for layer in [1,14]:
        def tensor(suffix,dtype):
            r=items[f'layers.{layer}.engram.{suffix}'];p=directory/r['file']
            assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
            return np.fromfile(p,dtype=dtype).reshape(r['shape'])
        for suffix,stem in [('q_weight','engram_q_norm'),('k_weight','engram_k_norm')]:
            writer.add_tensor(f'blk.{layer}.{stem}',tensor(suffix,'<u2'),raw_dtype=gguf.GGMLQuantizationType.BF16)
        expanded=expand_fp8(tensor('wkv.weight',np.uint8),tensor('wkv.scale',np.uint8))
        writer.add_tensor(f'blk.{layer}.engram_wkv',expanded,raw_dtype=gguf.GGMLQuantizationType.BF16)
    writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
    partial=destination.with_suffix('.partial');digest=sha(partial);partial.replace(destination)
    return dict(file=destination.name,bytes=destination.stat().st_size,sha256=digest,tensors=6,projection_source=manifest)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backbone',type=Path,required=True);p.add_argument('--engram',type=Path,required=True)
    p.add_argument('--projections',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--vocab',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    manifest_path=a.output/'package.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {'status':'building','files':[],'started':time.time()}
    def save():manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    def completed(path):
        if not path.exists():return False
        row=next((r for r in manifest['files'] if r['file']==path.name),None)
        assert row and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'unverified preexisting destination'
        return True
    source_config=json.loads((a.backbone/'config.json').read_text());c=source_config['text_config']
    ref=json.loads((a.reference/'engram-hash-reference.json').read_text())['config']
    token_map_path=a.reference/ref['token_map_file'];assert sha(token_map_path)==ref['token_map_sha256']
    token_map=np.fromfile(token_map_path,dtype='<u4').tolist();assert len(token_map)==c['vocab_size']
    assets={'version':1,'hash':{'compressed_vocab_size':ref['compressed_vocab_size'],'pad_compressed_id':ref['pad_compressed_id'],
        'token_map':token_map,'layers':[]},'rows':[]}
    for i,layer in enumerate(ref['layer_ids']):
        assets['hash']['layers'].append({'layer_id':layer,'rows':ref['num_embeddings'][i],'multipliers':ref['multipliers'][i],'primes':ref['primes'][i]})
        assets['rows'].append({'layer_id':layer,'row_count':ref['num_embeddings'][i],'file':f'engram-{layer}.affine2',
            'encoding':'affine2-g64-bf16','data_offset':0,'file_bytes':ref['num_embeddings'][i]*80})
    fields={'block_count':c['num_hidden_layers'],'vocab_size':c['vocab_size'],'context_length':c['max_position_embeddings'],
        'embedding_length':c['hidden_size'],'feed_forward_length':c['moe_intermediate_size'],
        'attention.head_count':c['num_attention_heads'],'attention.head_count_kv':1,'attention.key_length':c['head_dim'],
        'attention.value_length':c['head_dim'],'rope.dimension_count':c['qk_rope_head_dim'],'expert_count':c['n_routed_experts'],
        'expert_used_count':c['num_experts_per_tok']}
    common={f'deepseek41.{k}':record(f'deepseek41.{k}',4,v) for k,v in fields.items()}
    for key,value in [('general.architecture','deepseek41'),('general.name','DeepSeek-V4.1-Flash mixed-Q2 native'),
                      ('deepseek41.config',json.dumps(source_config)),('deepseek41.assets',json.dumps(assets))]:common[key]=record(key,8,value)
    vocab_records=header_parts(a.vocab)[0]
    common.update({k:v for k,v in vocab_records.items() if k.startswith('tokenizer.')})
    sources=sorted(a.backbone.glob('*.gguf'));assert len(sources)==5
    total=sum(header_parts(s)[2] for s in sources)+6;assert total==1044
    expected={Path(line.split()[1]).name:line.split()[0] for line in (a.backbone/'SHA256SUMS').read_text().splitlines()}
    needed=sum(s.stat().st_size for s in sources)+sum(x['file_bytes'] for x in assets['rows'])+700_000_000
    if not manifest['files']:assert shutil.disk_usage(a.output).free>needed+8_000_000_000,'insufficient free space'
    manifest.update(pid=os.getpid(),tokenizer_sha256=sha(a.backbone/'tokenizer.json'),assets=assets,status='building');save()
    for i,source in enumerate(sources):
        dest=a.output/f'DeepSeek-V4.1-Flash-MixedQ2-{i+1:05d}-of-00006.gguf'
        if completed(dest):continue
        extra=dict(common) if i==0 else {k:v for k,v in common.items() if not k.startswith('tokenizer.') and k!='deepseek41.assets'}
        extra.update({'split.no':record('split.no',2,i),'split.count':record('split.count',2,6),'split.tensors.count':record('split.tensors.count',5,total)})
        manifest['current']=dest.name;save()
        manifest['files'].append(copy_backbone(source,dest,extra,expected[source.name]));save()
    for i,layer in enumerate(ref['layer_ids']):
        dest=a.output/f'engram-{layer}.affine2'
        if completed(dest):continue
        manifest['current']=dest.name;save()
        source=a.engram/f'model-{47+i:05d}-of-00048.safetensors'
        manifest['files'].append(convert_rows(source,dest,layer,ref['num_embeddings'][i]));save()
    dest=a.output/'DeepSeek-V4.1-Flash-MixedQ2-00006-of-00006.gguf'
    if not completed(dest):
        manifest['current']=dest.name;save();manifest['files'].append(projection_shard(a.projections,dest,6,total));save()
    for name in ['config.json','tokenizer.json','tokenizer_config.json']:shutil.copyfile(a.backbone/name,a.output/name)
    manifest.update(status='complete',finished=time.time());manifest.pop('current',None);save()
    print('Native package complete',flush=True)

if __name__=='__main__':main()

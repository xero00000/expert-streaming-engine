#!/usr/bin/env python3
"""Repair staged GGUF padding without changing tensor payloads; resumable journal."""
import argparse
import json
import os
from pathlib import Path
import time
from deepseek_v41_package import copy_backbone,sha
p=argparse.ArgumentParser(description=__doc__);p.add_argument('package',type=Path);a=p.parse_args()
manifest_path=a.package/'package.json';journal=a.package/'alignment-pending.json'
manifest=json.loads(manifest_path.read_text());assert manifest['status'] in ['complete','realigning']
def atomic(path,value):
 temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)
def publish(record,verify=True):
 target=a.package/record['file'];temporary=a.package/record['temporary']
 if temporary.exists():
  if verify:assert sha(temporary)==record['entry']['sha256']
  temporary.replace(target)
 else:assert sha(target)==record['entry']['sha256']
 index=next(i for i,r in enumerate(manifest['files']) if r['file']==record['file'])
 manifest['files'][index]=record['entry'];atomic(manifest_path,manifest);journal.unlink(missing_ok=True)
if journal.exists():publish(json.loads(journal.read_text()))
manifest.update(status='realigning',pid=os.getpid());atomic(manifest_path,manifest)
for old in list(manifest['files']):
 if 'tensor_body_sha256' not in old:continue
 target=a.package/old['file'];temporary=target.with_suffix('.realigned.gguf')
 manifest['current']=old['file'];atomic(manifest_path,manifest)
 result=copy_backbone(target,temporary,{},old['sha256'])
 result.update(file=old['file'],source_sha256=old['source_sha256'],pre_alignment_sha256=old['sha256'],
               original_tensor_body_sha256=old['tensor_body_sha256'])
 record={'file':old['file'],'temporary':temporary.name,'entry':result}
 atomic(journal,record);publish(record,verify=False)
 print('Aligned and verified',old['file'],flush=True)
manifest.update(status='complete',alignment_finished=time.time());manifest.pop('current',None);atomic(manifest_path,manifest)
print('Package alignment complete',flush=True)

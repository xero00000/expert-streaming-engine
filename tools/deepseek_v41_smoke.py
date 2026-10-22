#!/usr/bin/env python3
"""Run one recorded, bounded native V4.1 generation check after staging completes."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import resource
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--package',type=Path,required=True)
p.add_argument('--binary',type=Path,required=True)
p.add_argument('--encoder',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
question=p.add_mutually_exclusive_group()
question.add_argument('--question',default='What is 2 + 2? Answer briefly.')
question.add_argument('--question-file',type=Path)
question.add_argument('--raw-prompt-file',type=Path)
p.add_argument('--logits-file',type=Path)
p.add_argument('--tokens',type=int,default=32)
p.add_argument('--prepare-only',action='store_true')
p.add_argument('--max-rss-mib',type=int,default=32768)
p.add_argument('--gpu-layers',type=int,default=41)
p.add_argument('--expert-vram-cache-mib',type=int,default=1024)
p.add_argument('--expert-vram-reserve-mib',type=int,default=768)
p.add_argument('--expert-ram-cache-mib',type=int,default=8192)
p.add_argument('--tensor-split',default='10,10,6')
p.add_argument('--cuda-visible-devices')
p.add_argument('--expert-device')
p.add_argument('--expert-layer-count',type=int,choices=range(41))
p.add_argument('--ubatch',type=int,choices=range(1,9),default=1)
p.add_argument('--direct-bf16-head',action='store_true',help='use the experimental F32-accumulating BF16 CUDA vocabulary kernel')
p.add_argument('--ram-retention',choices=('lru','frequency'),default='lru')
p.add_argument('--expert-trace',type=Path,help='append expert route and host wait events as JSONL')
p.add_argument('--early-router',choices=('off','probe','advice'),default='off',help='experimental early route forecast; only advice performs speculative storage I/O')
p.add_argument('--prefetch-mib',type=int,default=64,help='maximum speculative advice bytes per graph pass, in MiB')
p.add_argument('--pinned-ram',action='store_true',help='use a fixed CUDA-pinned expert RAM arena (at most 1 GiB)')
p.add_argument('--prefetch-confidence',type=float,default=0.85,help='minimum past precision on predicted VRAM misses before advising reads')
p.add_argument('--file-advice',choices=('normal','random'),default='normal',help='Linux page-cache access hint for expert sources')
p.add_argument('--threads',type=int,default=8)
p.add_argument('--load-workers',type=int,default=2)
p.add_argument('--tensor-override',action='append',default=[])
p.add_argument('--cache-retention',choices=['lru','frequency'],default='lru')
a=p.parse_args()
if a.question_file:a.question=a.question_file.read_text()
assert 1<=a.tokens<=256 and a.max_rss_mib>0
expected_encoder='502bdaec8a3fd88ebc24c4721a7038fbe42f2063c664638127056107920035c1'
assert hashlib.sha256(a.encoder.read_bytes()).hexdigest()==expected_encoder,'encoder differs from pinned reference'
spec=importlib.util.spec_from_file_location('v41_prompt_reference',a.encoder)
encoder=importlib.util.module_from_spec(spec);spec.loader.exec_module(encoder)
messages=[] if a.raw_prompt_file else [{'role':'user','content':a.question}]
prompt=a.raw_prompt_file.read_text() if a.raw_prompt_file else encoder.encode_messages(messages,thinking_mode='chat')
a.output.mkdir(parents=True,exist_ok=True)
prompt_path=a.output/'prompt.txt';prompt_path.write_text(prompt)
model=a.package/'DeepSeek-V4.1-Flash-MixedQ2-00001-of-00006.gguf'
command=[str(a.binary.resolve()),'-m',str(model.resolve()),'--cpu-moe','--defer-experts',
 '--expert-ram-cache-mib',str(a.expert_ram_cache_mib),'--expert-ram-staging-mib','256','--expert-sidecar-only',
 '--expert-vram-cache-mib',str(a.expert_vram_cache_mib),'--expert-vram-reserve-mib',str(a.expert_vram_reserve_mib),'--expert-prefill-staging-mib','0',
 '--expert-load-workers',str(a.load_workers),'--split-mode','layer','--split-mode-f32','-ngl',str(a.gpu_layers),'-ts',a.tensor_split,'-c','1024','-b','32','-ub',str(a.ubatch),'-t',str(a.threads),
 '--no-warmup','--no-context-shift','--simple-io','--no-display-prompt','--seed','41','--temp','0',
 '-n',str(a.tokens),'-f',str(prompt_path.resolve())]
for override in a.tensor_override:command.extend(['--override-tensor',override])
if a.logits_file:command.extend(['--save-all-logits',str(a.logits_file.resolve())])
environment={'GGML_CUDA_NO_MODEL_PINNED':'1'}
environment['ESE_EXPERT_CACHE_RETENTION']=a.cache_retention
environment['ESE_EXPERT_RAM_RETENTION']=a.ram_retention
if a.ubatch>1:
 environment.update(ESE_CUDA_MOE_BATCH_VECTOR='1',ESE_CUDA_DENSE_BATCH_VECTOR='1')
for name in ('ESE_CUDA_MOE_BATCH_VECTOR','ESE_CUDA_DENSE_BATCH_VECTOR','ESE_EVAL_FINAL_LOGIT_ONLY','ESE_V41_DIRECT_BF16_HEAD','ESE_V41_FULL_INDEXER'):
 if name in os.environ:environment[name]=os.environ[name]
if a.direct_bf16_head:environment['ESE_V41_DIRECT_BF16_HEAD']='1'
if a.expert_trace:environment['ESE_EXPERT_TRACE']=str(a.expert_trace.resolve())
environment['ESE_V41_EARLY_ROUTER']=a.early_router
environment['ESE_EXPERT_FUTURES_MIB']=str(a.prefetch_mib)
environment['ESE_EXPERT_RAM_PINNED']='1' if a.pinned_ram else '0'
environment['ESE_V41_PREFETCH_CONFIDENCE']=str(a.prefetch_confidence)
environment['ESE_EXPERT_FILE_ADVICE']=a.file_advice
if a.expert_device:environment['ESE_V41_EXPERT_DEVICE']=a.expert_device
if a.expert_layer_count is not None:environment['ESE_V41_EXPERT_LAYER_COUNT']=str(a.expert_layer_count)
if a.cuda_visible_devices:environment['CUDA_VISIBLE_DEVICES']=a.cuda_visible_devices
report={'status':'prepared','max_rss_mib':a.max_rss_mib,'command':command,'environment':environment,'messages':messages,'prompt':prompt,'encoder_sha256':expected_encoder}
report['prompt_source']='raw file' if a.raw_prompt_file else 'official chat encoder'
report_path=a.output/'run.json'
def save():report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
save()
if a.prepare_only:
 print('Prepared generation check; not executed')
 raise SystemExit
package=json.loads((a.package/'package.json').read_text())
assert package['status']=='complete' and len(package['files'])==8,'native package is not complete'
assert package.get('dense_repair',{}).get('tensors')==328,'original dense FP8 correction is required for this quant artifact'
for item in package['files']:
 assert (a.package/item['file']).stat().st_size==item['bytes'],'staged file size changed'
report['package_manifest_sha256']=hashlib.sha256((a.package/'package.json').read_bytes()).hexdigest()
report['validation']='completed verified-stage manifest plus current file sizes'
start=time.time();peak_rss=0;gpu_peaks={};last_gpu_sample=0;process_io={};cpu_seconds=0.0
with (a.output/'stdout.txt').open('wb') as out,(a.output/'stderr.txt').open('wb') as err:
 def child_limits():resource.setrlimit(resource.RLIMIT_CORE,(0,0))
 process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=out,stderr=err,preexec_fn=child_limits,env={**os.environ,**environment})
 report.update(status='running',pid=process.pid,started=start);save()
 while process.poll() is None:
  try:
   for line in Path(f'/proc/{process.pid}/status').read_text().splitlines():
    if line.startswith(('VmRSS:','VmHWM:')):peak_rss=max(peak_rss,int(line.split()[1])*1024)
   for line in Path(f'/proc/{process.pid}/io').read_text().splitlines():
    key,value=line.split(':');process_io[key]=max(process_io.get(key,0),int(value))
   fields=Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()
   cpu_seconds=max(cpu_seconds,(int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK'))
  except FileNotFoundError:pass
  if 'first_output_seconds' not in report and (a.output/'stdout.txt').stat().st_size:
   report['first_output_seconds']=time.time()-start
  if time.time()-last_gpu_sample>=2:
   last_gpu_sample=time.time()
   try:
    sample=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
    if sample.returncode==0:
     for line in sample.stdout.splitlines():
      device,used=[int(value.strip()) for value in line.split(',')]
      gpu_peaks[str(device)]=max(gpu_peaks.get(str(device),0),used)
   except (FileNotFoundError,subprocess.TimeoutExpired,ValueError):pass
  if peak_rss>a.max_rss_mib*(1<<20):
   report['memory_limit_exceeded']=True
   process.terminate()
   try:process.wait(timeout=10)
   except subprocess.TimeoutExpired:process.kill();process.wait()
   break
  time.sleep(1)
 peak_rss=max(peak_rss,resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024)
 report.update(process_io_sampled=process_io,cpu_seconds_sampled=cpu_seconds)
 report.update(status='memory_limit' if report.get('memory_limit_exceeded') else 'exited',returncode=process.returncode,wall_seconds=time.time()-start,peak_rss_bytes=peak_rss,peak_rss_method='live samples plus waited child high-water mark',gpu_peak_used_mib=gpu_peaks,gpu_memory_scope='sampled device totals, including other processes')
 save()
print(json.dumps({k:report[k] for k in ['status','returncode','wall_seconds','peak_rss_bytes']}))
raise SystemExit(process.returncode)

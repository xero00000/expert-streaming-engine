import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import deepseek_v41_package as package
from deepseek_v41_dense_repair import patch_tensor, file_hashes

class PackageTest(unittest.TestCase):
    def test_dense_repair_is_bounded_and_replayable(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'native.gguf'
            writer=package.gguf.GGUFWriter(path,'deepseek41')
            writer.add_tensor('before',np.arange(32,dtype=np.float32))
            writer.add_tensor('target',np.arange(32,dtype=np.float32))
            writer.add_tensor('after',np.arange(32,dtype=np.float32))
            writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
            before=path.read_bytes();header=package.header_parts(path)
            offset=header[3]+next(t['relative_offset'] for t in header[5] if t['name']=='target')
            replacement=np.full(32,41,dtype=np.float32).tobytes()
            for _ in range(2):
                patch_tensor(path,offset,replacement)
                self.assertEqual(path.read_bytes(),before[:offset]+replacement+before[offset+len(replacement):])
            digest,payload=file_hashes(path)
            self.assertEqual(digest,package.sha(path))
            self.assertEqual(payload,hashlib.sha256(path.read_bytes()[header[3]:]).hexdigest())
            with self.assertRaises(AssertionError):patch_tensor(path,len(before)-4,replacement)

    def test_reheader_preserves_tensor_body_and_types(self):
        with tempfile.TemporaryDirectory() as root:
            source=Path(root)/'source.gguf';dest=Path(root)/'native.gguf'
            writer=package.gguf.GGUFWriter(source,'deepseek4')
            writer.add_uint32('deepseek4.embedding_length',32)
            writer.add_tensor('blk.0.test',np.arange(128,dtype=np.float32).reshape(4,32))
            writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
            result=package.copy_backbone(source,dest,{'general.architecture':package.record('general.architecture',8,'deepseek41'),
                'deepseek41.config':package.record('deepseek41.config',8,'{"x":1}')},package.sha(source))
            src=package.header_parts(source);dst=package.header_parts(dest)
            self.assertEqual(src[5],dst[5]);self.assertEqual(dst[4]['general.architecture'],'deepseek41')
            self.assertNotIn('deepseek4.embedding_length',dst[4])
            self.assertEqual(source.read_bytes()[src[3]:],dest.read_bytes()[dst[3]:])
            self.assertEqual(package.sha(dest),result['sha256'])

    def test_misadvertised_alignment_is_repaired_and_resumable(self):
        with tempfile.TemporaryDirectory() as root:
            source=Path(root)/'bad.gguf';dest=Path(root)/'fixed.gguf'
            def info(name,width,offset):
                return package.string(name)+struct.pack('<IQIQ',1,width,0,offset)
            header=b'GGUF'+struct.pack('<IQQ',3,2,1)+package.record('general.architecture',8,'deepseek4')
            header+=info('scale',3,0)+info('matrix',32,12)
            header+=b'\0'*(-len(header)%32)
            first=np.arange(3,dtype=np.float32).tobytes();second=np.arange(32,dtype=np.float32).tobytes()
            source.write_bytes(header+first+second)
            result=package.copy_backbone(source,dest,{},package.sha(source))
            parsed=package.header_parts(dest);table={t['name']:t for t in parsed[5]}
            self.assertEqual(table['matrix']['relative_offset'],32)
            self.assertEqual(dest.read_bytes()[parsed[3]:],first+b'\0'*20+second)
            expected=dest.read_bytes();dest.with_suffix('.partial').write_bytes(expected[:parsed[3]+51]);dest.unlink()
            resumed=package.copy_backbone(source,dest,{},package.sha(source))
            self.assertEqual(dest.read_bytes(),expected);self.assertEqual(resumed['sha256'],result['sha256'])

    def test_affine2_copacking_is_byte_exact(self):
        with tempfile.TemporaryDirectory() as root:
            source=Path(root)/'source.safetensors';dest=Path(root)/'rows.affine2'
            rng=np.random.default_rng(41);n=37;parts=[rng.integers(0,256,(n,w),dtype=np.uint8) for w in [64,8,8]]
            header={};offset=0
            for suffix,data,dtype,columns in zip(['weight','scales','biases'],parts,['U32','BF16','BF16'],[16,4,4]):
                header['layers.1.engram.embed.'+suffix]={'dtype':dtype,'shape':[n,columns],'data_offsets':[offset,offset+data.nbytes]};offset+=data.nbytes
            raw=json.dumps(header).encode()
            source.write_bytes(struct.pack('<Q',len(raw))+raw+b''.join(x.tobytes() for x in parts))
            result=package.convert_rows(source,dest,1,n)
            self.assertEqual(dest.read_bytes(),np.concatenate(parts,axis=1).tobytes())
            self.assertEqual(package.sha(dest),result['sha256'])

    def test_fp8_projection_expansion_matches_torch(self):
        import torch
        rng=np.random.default_rng(410)
        weights=rng.integers(0,256,(96,64),dtype=np.uint8)
        weights[(weights==127)|(weights==255)]=0
        scales=rng.integers(123,132,(3,2),dtype=np.uint8)
        w=torch.from_numpy(weights.copy()).view(torch.float8_e4m3fn).float()
        s=torch.from_numpy(scales.copy()).view(torch.float8_e8m0fnu).float().repeat_interleave(32,0).repeat_interleave(32,1)
        expected=(w*s).bfloat16().view(torch.int16).numpy().view(np.uint16)
        np.testing.assert_array_equal(package.expand_fp8(weights,scales),expected)

if __name__=='__main__':unittest.main()

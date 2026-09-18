"""CPU-only checks for release tooling; synthetic inputs are not benchmarks."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_keys', ROOT / 'tools/prepare_keys.py')
keys_tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(keys_tool)


class ReleaseTests(unittest.TestCase):
    def test_keys_come_from_files_and_reject_incomplete_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for key in ['00000000/btest_000002', '00000000/btest_000001']:
                p = root / 'shapenet' / key / 'noisy.npy'
                p.parent.mkdir(parents=True)
                np.save(p, np.zeros((2, 3), np.float32))
            self.assertEqual(keys_tool.discover(root, 2), [f'shapenet/00000000/btest_{i:06d}' for i in [1, 2]])
            with self.assertRaises(ValueError):
                keys_tool.discover(root, 200)

    def test_fusion_preserves_indices_and_package_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            key = 'shapenet/00000000/btest_000001'
            keys = root / 'keys.txt';keys.write_text(key + '\n')
            x = np.arange(150000, dtype=np.float32).reshape(50000, 3) / 150000
            arrays = {'input': x, 'a': x + .010, 'b': x + .012, 'c': x + .080, 'vm': x + .020}
            for name, arr in arrays.items():
                p = root / name / key;p.mkdir(parents=True)
                np.save(p / ('noisy.npy' if name == 'input' else 'denoised.npy'), arr)
            fused = root / 'fused'
            subprocess.run([sys.executable, str(ROOT/'fusion/fuse_global_m3i1_vf_n01.py'), '--noisy-root', str(root/'input'), '--key-list', str(keys), '--expected-count', '1', '--output-root', str(fused), '--mamba-48k', str(root/'a'), '--mamba-60k', str(root/'b'), '--ipfn-1024', str(root/'c'), '--vm-fixed', str(root/'vm')], check=True, capture_output=True)
            prediction = np.load(fused/key/'denoised.npy')
            np.testing.assert_allclose(prediction, x + .0118, rtol=0, atol=2e-7)
            self.assertEqual(prediction.dtype, np.float32)
            out = root/'result.zip';audit=root/'audit.json'
            subprocess.run([sys.executable, str(ROOT/'code/package_result.py'), '--pred-root', str(fused), '--zip-path', str(out), '--audit-json', str(audit), '--expected-count', '1'], check=True, capture_output=True)
            with zipfile.ZipFile(out) as z:
                self.assertEqual(z.namelist(), [key+'/denoised.npy'])
                self.assertEqual(z.read(key+'/denoised.npy'), (fused/key/'denoised.npy').read_bytes())
            self.assertEqual(json.loads(audit.read_text())['status'], 'passed')


if __name__ == '__main__':
    unittest.main()

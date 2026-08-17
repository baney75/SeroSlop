import unittest
from benchmark.m6.p7_operational import verify_member, verify_receipts, build_source_lock, P7_PARENT, materialize_taste
from pathlib import Path
import tempfile
from PIL import Image
from benchmark.m6.p7_operational import _materialize_taste_rows, reopen_taste, atomic_write_bundle

def _row():
 return {"card":"a"*64,"container":"b"*64,"member":"x.png","encodedBytesSha256":"c"*64,"decodedRgbSha256":"d"*64,"dhash64":"e"*16,"width":4,"height":4,"pillow":"11.3.0"}

class P7OperationalTests(unittest.TestCase):
 def test_member_receipt_and_bundle(self):
    receipt = verify_member(_row(), source_scope="aigenimages2026-train")
    summary = verify_receipts([receipt], expected=1)
    self.assertIs(summary["h3PixelsRead"], False)
    self.assertTrue(summary["expandedSha256"])

 def test_member_rejects_wrong_pillow(self):
    row = _row(); row["pillow"] = "10.0"
    with self.assertRaises(ValueError): verify_member(row, source_scope="x")

 def test_receipt_shortage(self):
    with self.assertRaises(ValueError): verify_receipts([], expected=1)

 def test_taste_rejects_duplicate_asset_id(self):
    root = Path(tempfile.mkdtemp()); (root / "assets.parquet").write_bytes(b"bad")
    rows = [{"asset_id": 1, "model": "GPT Image 1.5", "image_url": "https://x", "track": "descriptions", "image_path": "a", "memberPath": "a"}] * 2
    with self.assertRaises(ValueError): materialize_taste(cache_root=root, output=root / "out", rows=rows)

 def test_taste_output_requires_absent_target(self):
    root = Path(tempfile.mkdtemp()); target = root / "out"; target.mkdir()
    with self.assertRaises(FileExistsError):
      materialize_taste(cache_root=Path("benchmark/data/m6-frontier-cache/taste"), output=target)

 def _fixture(self):
    root = Path(tempfile.mkdtemp()); (root / "images").mkdir(); (root / "assets.parquet").write_bytes(b"fixture")
    rows=[]
    for i in range(4):
      name=f"images/{i}.png"; Image.new("RGB", (4,4), (i,20,40)).save(root/name)
      rows.append({"asset_id":i+1,"model":"GPT Image 1.5","track":"descriptions","memberPath":name})
    return root, rows

 def test_private_fixture_deterministic_and_receipt_hash(self):
    root, rows = self._fixture(); a=root/"a"; b=root/"b"
    _materialize_taste_rows(root=root, output=a, rows=rows, expected_count=4, enforce_production=False)
    _materialize_taste_rows(root=root, output=b, rows=rows, expected_count=4, enforce_production=False)
    self.assertEqual((a/"taste-receipts.json").read_bytes(), (b/"taste-receipts.json").read_bytes())
    import json, hashlib
    values=json.loads((a/"taste-receipts.json").read_bytes())
    for rec in values:
      claimed=rec.pop("receiptSha256"); self.assertEqual(claimed, hashlib.sha256((__import__('benchmark.m6.p7_operational',fromlist=['canonical_json']).canonical_json(rec))).hexdigest())
    self.assertEqual(reopen_taste(a, enforce_production=False)["rows"], 4)

 def test_private_fixture_rejects_paths_symlink_duplicate_and_mutation(self):
    root, rows = self._fixture()
    bad=list(rows); bad[0]={**bad[0],"memberPath":"/tmp/x.png"}
    with self.assertRaises(ValueError): _materialize_taste_rows(root=root, output=root/"bad1", rows=bad, expected_count=4, enforce_production=False)
    bad=list(rows); bad[0]={**bad[0],"memberPath":"images/../x.png"}
    with self.assertRaises(ValueError): _materialize_taste_rows(root=root, output=root/"bad2", rows=bad, expected_count=4, enforce_production=False)
    bad=list(rows); bad[1]={**bad[1],"asset_id":1}
    with self.assertRaises(ValueError): _materialize_taste_rows(root=root, output=root/"bad3", rows=bad, expected_count=4, enforce_production=False)

 def test_fixture_symlink_and_tampered_reopen(self):
    root, rows = self._fixture(); (root/"images/link.png").symlink_to(root/"images/0.png")
    bad=[*rows]; bad[0]={**bad[0],"memberPath":"images/link.png"}
    with self.assertRaises(ValueError): _materialize_taste_rows(root=root, output=root/"bad", rows=bad, expected_count=4, enforce_production=False)
    out=root/"ok"; _materialize_taste_rows(root=root, output=out, rows=rows, expected_count=4, enforce_production=False)
    data=(out/"taste-summary.json").read_bytes(); (out/"taste-summary.json").write_bytes(data+b" ")
    with self.assertRaises(ValueError): reopen_taste(out, enforce_production=False)

 def test_reopen_rejects_symlinked_artifact_and_forged_canonical_receipt(self):
    import hashlib, json
    from benchmark.m6.p7_operational import canonical_json
    root, rows = self._fixture(); out=root/"ok"
    _materialize_taste_rows(root=root, output=out, rows=rows, expected_count=4, enforce_production=False)
    receipt_path=out/"taste-receipts.json"; original=receipt_path.read_bytes(); receipt_path.unlink(); receipt_path.symlink_to(root/"assets.parquet")
    with self.assertRaises(ValueError): reopen_taste(out, enforce_production=False)
    receipt_path.unlink(); receipt_path.write_bytes(original)
    receipts=json.loads(original); body=dict(receipts[0]); body.pop("receiptSha256"); body["publisher"]="forged"; body["receiptSha256"]=hashlib.sha256(canonical_json(body)).hexdigest(); receipts[0]=body
    receipt_path.write_bytes(canonical_json(receipts))
    summary=json.loads((out/"taste-summary.json").read_bytes()); summary["expandedSha256"]=verify_receipts(receipts)["expandedSha256"]; (out/"taste-summary.json").write_bytes(canonical_json(summary))
    with self.assertRaisesRegex(ValueError, "immutable binding"): reopen_taste(out, enforce_production=False)

 def test_source_lock_api_remains_disabled(self):
    with self.assertRaisesRegex(RuntimeError, "source-lock disabled"):
      build_source_lock(source_commit="f"*40, source_tree="1"*40, receipts={"x":"2"*64})

 def test_atomic_after_rename_rollback(self):
    root=Path(tempfile.mkdtemp()); out=root/"out"
    with self.assertRaises(RuntimeError): atomic_write_bundle(out,{"x":b"x"},failure_hook=lambda _: (_ for _ in ()).throw(RuntimeError("boom")))
    self.assertFalse(out.exists())

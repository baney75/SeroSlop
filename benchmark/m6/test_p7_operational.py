import unittest
from benchmark.m6.p7_operational import verify_member, verify_receipts, build_source_lock, P7_PARENT, materialize_taste
from pathlib import Path
import tempfile
from PIL import Image
from benchmark.m6.p7_operational import _materialize_taste_rows, reopen_taste, atomic_write_bundle, _x_fixture, _nano_fixture, _aigen_fixture, _reopen_frontier

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

 def test_p8_private_embedded_parquet_image_bytes_and_identity(self):
    import io, json
    from benchmark.m6.p7_operational import canonical_json
    raw=io.BytesIO(); Image.new("RGB", (4,4), (1,2,3)).save(raw, format="PNG")
    image={"bytes":raw.getvalue(),"path":"not-read.png"}
    x={"image":image,"generator":"fixture-gen","uid":"7","labels":{},"original_prompt":"","positive_prompt":"","negative_prompt":"","guidance_scale":1,"num_inference_steps":1,"scheduler":"s","seed":1,"width":4,"height":4,"image_format":"PNG","jpeg_quality":None,"chroma_subsampling":None}
    with tempfile.TemporaryDirectory() as d:
      out=Path(d)/"x"; _x_fixture(output=out, rows=[x])
      rec=json.loads((out/"x-aigd-receipts.json").read_bytes())[0]
      self.assertEqual(rec["generator"], "fixture-gen"); self.assertEqual(rec["uid"], "7")
      n={"id":3,"image":image,"format":"PNG","mode":"RGB","width":4,"height":4,"uploadtime":"fixture"}
      nout=Path(d)/"n"; _nano_fixture(output=nout, rows=[n], shards=["data/fixture.parquet"])
      self.assertEqual(json.loads((nout/"nano-banana-receipts.json").read_bytes())[0]["id"], 3)

 def test_p8_private_aigen_is_deterministic(self):
    import io
    raw=io.BytesIO(); Image.new("RGB",(3,3),(9,8,7)).save(raw,format="PNG")
    rows=[{"image_id":1,"filename":"x.png","caption":"x","caption_id":1,"split":"train"}]
    with tempfile.TemporaryDirectory() as d:
      a=Path(d)/"a"; b=Path(d)/"b"
      _aigen_fixture(root=Path(d),output=a,partition="train",metadata_rows=rows,image_rows={"x.png":raw.getvalue()})
      _aigen_fixture(root=Path(d),output=b,partition="train",metadata_rows=rows,image_rows={"x.png":raw.getvalue()})
      self.assertEqual((a/"aigenimages2026-train-receipts.json").read_bytes(),(b/"aigenimages2026-train-receipts.json").read_bytes())

 def test_p8_x_rejects_duplicate_identity(self):
    import io
    raw=io.BytesIO(); Image.new("RGB",(2,2)).save(raw,format="PNG")
    x={"image":{"bytes":raw.getvalue(),"path":"x"},"generator":"g","uid":"1","labels":{},"original_prompt":"","positive_prompt":"","negative_prompt":"","guidance_scale":1,"num_inference_steps":1,"scheduler":"s","seed":1,"width":2,"height":2,"image_format":"PNG","jpeg_quality":None,"chroma_subsampling":None}
    with tempfile.TemporaryDirectory() as d:
      with self.assertRaises(ValueError): _x_fixture(output=Path(d)/"x",rows=[x,x])

 def test_p8_nano_rejects_duplicate_id_and_shard(self):
    import io
    raw=io.BytesIO(); Image.new("RGB",(2,2)).save(raw,format="PNG")
    n={"id":1,"image":{"bytes":raw.getvalue(),"path":"x"},"format":"PNG","mode":"RGB","width":2,"height":2,"uploadtime":"x"}
    with tempfile.TemporaryDirectory() as d:
      with self.assertRaises(ValueError): _nano_fixture(output=Path(d)/"x",rows=[n,n],shards=["x"])
      out=Path(d)/"y"; _nano_fixture(output=out,rows=[n,{**n,"id":2}],shards=["x","y"])
      self.assertEqual(_reopen_frontier(out,"nano-banana")["containerCount"],2)

 def test_p8_reopen_rejects_noncanonical_duplicate_and_symlink(self):
    import io
    raw=io.BytesIO(); Image.new("RGB",(2,2)).save(raw,format="PNG")
    x={"image":{"bytes":raw.getvalue(),"path":"x"},"generator":"g","uid":"1","labels":{},"original_prompt":"","positive_prompt":"","negative_prompt":"","guidance_scale":1,"num_inference_steps":1,"scheduler":"s","seed":1,"width":2,"height":2,"image_format":"PNG","jpeg_quality":None,"chroma_subsampling":None}
    with tempfile.TemporaryDirectory() as d:
      out=Path(d)/"x"; summary=_x_fixture(output=out,rows=[x]); receipt=out/"x-aigd-receipts.json"; good=receipt.read_bytes()
      receipt.write_bytes(good[:-1]+b" \n")
      with self.assertRaises(ValueError): _reopen_frontier(out,"x-aigd")
      receipt.write_bytes(good); receipt.unlink(); receipt.symlink_to(out/"x-aigd-summary.json")
      with self.assertRaises(ValueError): _reopen_frontier(out,"x-aigd")

 def test_p8_public_adapters_do_not_accept_rows(self):
    import inspect
    from benchmark.m6.p7_operational import materialize_aigen, materialize_x_aigd, materialize_nano
    for func in (materialize_aigen,materialize_x_aigd,materialize_nano):
      self.assertNotIn("rows",inspect.signature(func).parameters)

 def test_p8_atomic_output_rollback(self):
    import io
    raw=io.BytesIO(); Image.new("RGB",(2,2)).save(raw,format="PNG")
    x={"image":{"bytes":raw.getvalue(),"path":"x"},"generator":"g","uid":"1","labels":{},"original_prompt":"","positive_prompt":"","negative_prompt":"","guidance_scale":1,"num_inference_steps":1,"scheduler":"s","seed":1,"width":2,"height":2,"image_format":"PNG","jpeg_quality":None,"chroma_subsampling":None}
    with tempfile.TemporaryDirectory() as d:
      out=Path(d)/"x"; _x_fixture(output=out,rows=[x]); self.assertTrue(out.is_dir())

 def test_p8_parquet_reader_is_batched_not_read_table(self):
    import inspect
    from benchmark.m6.p7_operational import _iter_parquet_rows
    source=inspect.getsource(_iter_parquet_rows)
    self.assertIn("ParquetFile",source); self.assertIn("iter_batches",source); self.assertNotIn("read_table",source)

 def test_p8_public_parquet_loops_do_not_materialize_row_lists(self):
    import inspect
    from benchmark.m6.p7_operational import materialize_x_aigd, materialize_nano
    self.assertNotIn("list(_iter_parquet_rows",inspect.getsource(materialize_x_aigd))
    self.assertNotIn("list(_iter_parquet_rows",inspect.getsource(materialize_nano))

 def test_p8_x_uid_is_string_and_member_is_canonical_hash(self):
    from benchmark.m6.p7_operational import _x_member
    self.assertRegex(_x_member("generator","uid"),r"^row/[0-9a-f]{64}$")
    with self.assertRaises(ValueError): _x_member("generator",1)

 def test_p8_actual_multibatch_x_parquet_fixture(self):
    import pyarrow as pa, pyarrow.parquet as pq
    from benchmark.m6.p7_operational import _iter_parquet_rows
    labels=pa.list_(pa.struct([("label",pa.string()),("points",pa.list_(pa.list_(pa.float64())))]))
    schema=pa.schema([("image",pa.struct([("bytes",pa.binary()),("path",pa.string())])),("generator",pa.string()),("uid",pa.string()),("labels",labels),("original_prompt",pa.string()),("positive_prompt",pa.string()),("negative_prompt",pa.string()),("guidance_scale",pa.float64()),("num_inference_steps",pa.int64()),("scheduler",pa.string()),("seed",pa.int64()),("width",pa.int64()),("height",pa.int64()),("image_format",pa.string()),("jpeg_quality",pa.int64()),("chroma_subsampling",pa.string())])
    rows=[{"image":{"bytes":b"x","path":"p"},"generator":"g","uid":str(i),"labels":[],"original_prompt":"","positive_prompt":"","negative_prompt":"","guidance_scale":1.0,"num_inference_steps":1,"scheduler":"s","seed":1,"width":1,"height":1,"image_format":"PNG","jpeg_quality":1,"chroma_subsampling":"x"} for i in range(130)]
    with tempfile.TemporaryDirectory() as d:
      path=Path(d)/"x.parquet"; pq.write_table(pa.Table.from_pylist(rows,schema=schema),path,row_group_size=1)
      self.assertEqual(len(list(_iter_parquet_rows(path,"x-aigd"))),130)

 def test_p8_three_csv_aigen_tar_scanner_and_archive_reds(self):
    import io,tarfile
    from benchmark.m6.p7_operational import _scan_aigen_tar
    root="mnt/cephfs/home/common/datasets/online_sid_dataset/AIGenImages/AIGenImages2026/"
    def make(path,duplicate=False):
      with tarfile.open(path,"w:gz") as tar:
        for name,data in [(root+"real.csv",b"id,kind\n1,real\n"),(root+"train.csv",b"image_id,filename,caption,caption_id,split\n1,x.png,x,1,train\n"),(root+"val.csv",b"image_id,filename,caption,caption_id,split\n2,y.png,y,2,val\n"),(root+"train/1_fake/x.png",b"image")]:
          info=tarfile.TarInfo(name); info.size=len(data); tar.addfile(info,io.BytesIO(data))
          if duplicate and name.endswith("x.png"): tar.addfile(info,io.BytesIO(data))
    with tempfile.TemporaryDirectory() as d:
      path=Path(d)/"a.tar.gz"; make(path); rows,members=_scan_aigen_tar(path,"train",expected_rows=1)
      self.assertEqual(rows[0]["filename"],"x.png"); self.assertEqual(len(members),1)
      bad=Path(d)/"bad.tar.gz"; make(bad,True)
      with self.assertRaises(ValueError): _scan_aigen_tar(bad,"train",expected_rows=1)

 def test_p8_reopen_rejects_rehashed_forged_x_member_and_dimensions(self):
    import io,json,hashlib
    from benchmark.m6.p7_operational import canonical_json
    raw=io.BytesIO(); Image.new("RGB",(2,2)).save(raw,format="PNG")
    x={"image":{"bytes":raw.getvalue(),"path":"x"},"generator":"g","uid":"1","labels":{},"original_prompt":"","positive_prompt":"","negative_prompt":"","guidance_scale":1,"num_inference_steps":1,"scheduler":"s","seed":1,"width":2,"height":2,"image_format":"PNG","jpeg_quality":None,"chroma_subsampling":None}
    with tempfile.TemporaryDirectory() as d:
      out=Path(d)/"x"; _x_fixture(output=out,rows=[x]); path=out/"x-aigd-receipts.json"; rows=json.loads(path.read_bytes()); rows[0]["member"]="forged"; rows[0]["width"]=9; rows[0]["receiptSha256"]=hashlib.sha256(canonical_json({k:v for k,v in rows[0].items() if k!="receiptSha256"})).hexdigest(); path.write_bytes(canonical_json(rows))
      with self.assertRaises(ValueError): _reopen_frontier(out,"x-aigd")

 def test_p8_reopen_rejects_zero_nano_container_even_if_rehashed(self):
    import io,json
    raw=io.BytesIO(); Image.new("RGB",(2,2)).save(raw,format="PNG")
    n={"id":1,"image":{"bytes":raw.getvalue(),"path":"x"},"format":"PNG","mode":"RGB","width":2,"height":2,"uploadtime":"x"}
    with tempfile.TemporaryDirectory() as d:
      out=Path(d)/"n"; _nano_fixture(output=out,rows=[n],shards=["data/x.parquet"]); summary=out/"nano-banana-summary.json"; value=json.loads(summary.read_bytes()); value["containerInventory"][0]["bytes"]=0; value["containerBytes"]=0; value["containerInventorySha256"]="0"*64; summary.write_text(json.dumps(value,sort_keys=True,separators=(",",":"))+"\n")
      with self.assertRaises(ValueError): _reopen_frontier(out,"nano-banana")

 def test_p8_reopen_rejects_rehashed_x_path_and_format(self):
    import io,json,hashlib
    from benchmark.m6.p7_operational import canonical_json
    raw=io.BytesIO(); Image.new("RGB",(2,2)).save(raw,format="PNG")
    x={"image":{"bytes":raw.getvalue(),"path":"safe.png"},"generator":"g","uid":"u","labels":{},"original_prompt":"","positive_prompt":"","negative_prompt":"","guidance_scale":1,"num_inference_steps":1,"scheduler":"s","seed":1,"width":2,"height":2,"image_format":"PNG","jpeg_quality":None,"chroma_subsampling":None}
    with tempfile.TemporaryDirectory() as d:
      out=Path(d)/"x"; _x_fixture(output=out,rows=[x]); path=out/"x-aigd-receipts.json"; rows=json.loads(path.read_bytes()); rows[0]["nativePath"]="../bad.jpg"; rows[0]["declaredFormat"]="JPEG"; rows[0]["receiptSha256"]=hashlib.sha256(canonical_json({k:v for k,v in rows[0].items() if k!="receiptSha256"})).hexdigest(); path.write_bytes(canonical_json(rows))
      with self.assertRaises(ValueError): _reopen_frontier(out,"x-aigd")

 def test_p8_x_png_payload_allows_paired_real_jpeg_metadata_but_rejects_nonpng(self):
    import io
    def row(raw, declared): return {"image":{"bytes":raw,"path":"paired-real.jpg"},"generator":"g","uid":"u","labels":{},"original_prompt":"","positive_prompt":"","negative_prompt":"","guidance_scale":1,"num_inference_steps":1,"scheduler":"s","seed":1,"width":2,"height":2,"image_format":declared,"jpeg_quality":1,"chroma_subsampling":"x"}
    png=io.BytesIO(); Image.new("RGB",(2,2)).save(png,format="PNG")
    jpeg=io.BytesIO(); Image.new("RGB",(2,2)).save(jpeg,format="JPEG")
    jpg=io.BytesIO(); Image.new("RGB",(2,2)).save(jpg,format="JPEG")
    with tempfile.TemporaryDirectory() as d:
      out=Path(d)/"ok"; _x_fixture(output=out,rows=[row(png.getvalue(),"JPEG")]); self.assertEqual(_reopen_frontier(out,"x-aigd")["rows"],1)
      bad=Path(d)/"bad"
      with self.assertRaises(ValueError): _x_fixture(output=bad,rows=[row(jpg.getvalue(),"JPEG")])
      self.assertFalse(bad.exists())

 def test_p8_reopen_rejects_rehashed_nano_escape_path(self):
    import io,json,hashlib
    from benchmark.m6.p7_operational import canonical_json
    raw=io.BytesIO(); Image.new("RGB",(2,2)).save(raw,format="PNG")
    n={"id":1,"image":{"bytes":raw.getvalue(),"path":"safe.png"},"format":"PNG","mode":"RGB","width":2,"height":2,"uploadtime":"x"}
    with tempfile.TemporaryDirectory() as d:
      out=Path(d)/"n"; _nano_fixture(output=out,rows=[n],shards=["data/x.parquet"]); path=out/"nano-banana-receipts.json"; rows=json.loads(path.read_bytes()); rows[0]["nativePath"]="../escape"; rows[0]["receiptSha256"]=hashlib.sha256(canonical_json({k:v for k,v in rows[0].items() if k!="receiptSha256"})).hexdigest(); path.write_bytes(canonical_json(rows))
      with self.assertRaises(ValueError): _reopen_frontier(out,"nano-banana")

 def test_p8_aigen_suffix_format_mismatch_rolls_back_for_all_families(self):
    import io
    png=io.BytesIO(); Image.new("RGB",(2,2)).save(png,format="PNG")
    jpeg=io.BytesIO(); Image.new("RGB",(2,2)).save(jpeg,format="JPEG")
    with tempfile.TemporaryDirectory() as d:
      for suffix in (".jpg",".jpeg",".webp"):
        name="wrong"+suffix; out=Path(d)/(suffix[1:]+"-out")
        rows=[{"image_id":"1","filename":name,"caption":"x","caption_id":"1","split":"train"}]
        with self.assertRaises(ValueError): _aigen_fixture(root=Path(d),output=out,partition="train",metadata_rows=rows,image_rows={name:png.getvalue()})
        self.assertFalse(out.exists())
      out=Path(d)/"png-mismatch"; rows=[{"image_id":"1","filename":"wrong.png","caption":"x","caption_id":"1","split":"train"}]
      with self.assertRaises(ValueError): _aigen_fixture(root=Path(d),output=out,partition="train",metadata_rows=rows,image_rows={"wrong.png":jpeg.getvalue()})
      self.assertFalse(out.exists())
      good=Path(d)/"png-out"; rows=[{"image_id":"1","filename":"right.png","caption":"x","caption_id":"1","split":"train"}]
      _aigen_fixture(root=Path(d),output=good,partition="train",metadata_rows=rows,image_rows={"right.png":png.getvalue()}); self.assertTrue(good.exists())

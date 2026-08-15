# Packaged detector

`prooflens-cf384.onnx` is the exact FP32 artifact described by the repository root [model-lock.json](../model-lock.json).

```text
bytes    87,442,080
sha256   a994b1bd4d0323909b2b308db848bf668fd00e2f02c8973ec546c400efe2dc47
input    pixel_values [N,3,384,384] float32
output   logits [N,1] float32
license  MIT
```

The build fails if the byte count or SHA-256 changes.

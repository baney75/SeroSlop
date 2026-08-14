# Packaged detector

`prooflens-cf384.onnx` is the exact FP32 artifact described by the repository root [model-lock.json](../model-lock.json).

```text
bytes    87,442,080
sha256   941e3914c075a735db5795e897b71c1d8b2f6b7c2cf2cb7777d0a6999aa02e6c
input    pixel_values [N,3,384,384] float32
output   logits [N,1] float32
license  MIT
```

The build fails if the byte count or SHA-256 changes.

# Packaged detector

`prooflens-cf384.onnx` is the exact FP32 artifact described by the repository root [model-lock.json](../model-lock.json).

```text
bytes    87,442,080
sha256   29545a1da0cfe2bf0149448334fd45a21f48074c57296db3b84437dd66f80a43
input    pixel_values [N,3,384,384] float32
output   logits [N,1] float32
license  MIT
```

The build fails if the byte count or SHA-256 changes.

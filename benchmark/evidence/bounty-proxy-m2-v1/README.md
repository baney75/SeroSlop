# M2 fixed-threshold H3 Met / TASTE submission proxy

This directory is reserved for a reproducible, score-blind manifest freeze and
an offline browser-authoritative scoring receipt. It is a submission proxy, not
bounty proof, training evidence, or a claim of provider approval.

`bounty-proxy:freeze` reads only H3 and TASTE metadata plus the fixed model
files. It does not open selected images or run inference. TASTE selects 150 of
each four 161-row model groups using a fixed SHA-256 order. Those synthetic
labels are publisher assertions, not an independent origin claim.

After the frozen packet is committed and public CI passes,
`bounty-proxy:verify-inputs` hashes the exact 1,200 selected image files.
`bounty-proxy:browser` then runs the packaged extension's `PL_INFER` path in a
clean Chrome profile after Chrome is offline. It refuses post-cutoff HTTP(S)
requests. The decision is inclusive: an unrounded display score at or above
`0.65` is flagged. The equivalent fixed raw probability threshold is
`0.27019907955040323`.

The panel contains 600 reserved Met Open Access real images and 600 TASTE
synthetic images, 150 from each declared model group. It is an original-view
submission proxy. It does not reveal or reproduce the maintainer's private
benchmark.

The completed clean-profile Chrome run used WebGPU for all 1,200 images. At
the inclusive 65.0/100 threshold it produced 85.83% balanced accuracy, 77.67%
real-image recall, and 94.00% synthetic-image recall (TN 466, FP 134, TP 564,
FN 36). Chrome was offline before inference, and the run recorded no HTTP or
HTTPS request after that cutoff.

`verified-inputs/` contains the exact input hashes. `results/` contains one
prediction row per frozen input and the canonical result receipt. Run
`npm run check:bounty-proxy-results` to recompute the metrics and receipt
digests and to recheck the recorded public Quality run. These results clear the
published 75% threshold on this proxy; they are not a claim about the private
benchmark or bounty acceptance.

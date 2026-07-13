# LDC: Learned Distance Conditioning

## Why this direction

The strongest verified improvements came from the velocity field: CVM-002
(`stage_velocity + deep supervision`) is the best online model at 75.51, while
purely wider/multi-scale models, pointwise gates, stronger noise, attention and
inference tricks did not transfer reliably.

LDC therefore keeps the complete CVM-002 model and adds only two low-risk ideas:

1. Condition each velocity encoder on the predicted remaining distance and its
   module stage using a zero-initialized FiLM adapter.
2. Unroll SPCF training for two iterations, exactly matching `tot_its=2` during
   inference instead of training one large step and inferring with two small
   steps.
3. Keep the pretrained velocity trunk frozen and explicitly re-enable gradients
   only for the new condition adapters. This is required because Jittor's
   `Module.eval()` freezes parameters during SPCF training.

## Safety properties

- Initialization loads the full CVM-002 SPCF checkpoint.
- The new condition projection is zero-initialized; before fine-tuning, LDC and
  CVM-002 produce identical velocity outputs (`max_abs_diff = 0`).
- Fine-tuning uses `lr=1e-5`.
- Only the distance head and zero-initialized LDC adapters are trainable; the
  pretrained velocity trunk remains frozen.
- The original checkpoint is copied to `checkpoint_baseline.pkl` as an explicit
  fallback. LDC selects its own best checkpoint because the matched-unroll loss
  is not numerically comparable with the historical single-unroll loss.

## Success criterion

- Primary: local2 exceeds 72.77 without materially sacrificing either CD or P2S.
- Secondary: a changed CD/P2S Pareto may still justify one online submission,
  because local2 differences below 0.1 have historically been unreliable.

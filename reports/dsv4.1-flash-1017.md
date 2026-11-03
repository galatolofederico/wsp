# DeepSeek V4.1 Flash 1017 — blind weight-space task identification

**Date:** 3 November 2026  
**Model:** DeepSeek V4.1 Flash 1017, served as `deepseek-flash`  
**Endpoint:** `https://api.deepseek.com`  
**Harness:** `wsp-read-harness/wsp_read.py`  
**Raw runs:** `outputs/dsv4.1-flash-1017-{cartpole,flappy,ballbeam,catch}.json`

## Summary

DeepSeek V4.1 Flash 1017 identified **3 of 4** anonymized checkpoints correctly
(**75%**, versus 25% chance on a four-way choice). It correctly recognised
CartPole, Flappy and Catch, and misclassified **BallBeam as CartPole** with 88%
confidence.

The result is best described as a **systematic confusion between two tasks with
the same observation layout**, not a random failure. CartPole and BallBeam both
use `[position, velocity, angle, angular velocity]` observations and symmetric
left/right actions. The model found the angle/angular-velocity signature in the
BallBeam weights, interpreted the angle as a pole angle, and concluded
CartPole. Its own explanation of why it rejected BallBeam is exactly the
assumption that the BallBeam checkpoint violated.

All four runs used thinking mode at maximum reasoning effort, with no
safetensors metadata, no release-description file, and no weight-text
truncation. Each prompt contained the complete checkpoint (8 tensors, 51,203
parameters) and the same four candidate tasks.

## Results

| Ground truth | Predicted | Correct | Confidence | Prompt tokens | Completion tokens | Reasoning tokens | Total tokens |
|---|---|:---:|---:|---:|---:|---:|---:|
| CartPole | CartPole | yes | 82% | 328,526 | 48,478 | 47,477 | 377,004 |
| Flappy | Flappy | yes | 87% | 323,298 | 47,511 | 46,830 | 370,809 |
| BallBeam | CartPole | **no** | 88% | 329,158 | 119,986 | 119,330 | 449,144 |
| Catch | Catch | yes | 88% | 327,445 | 54,369 | 53,328 | 381,814 |
| **Total** | | **3/4** | | **1,308,427** | **270,344** | **266,965** | **1,578,771** |

Confusion matrix:

| Truth \ Prediction | CartPole | Flappy | BallBeam | Catch |
|---|:---:|:---:|:---:|:---:|
| CartPole | **1** | 0 | 0 | 0 |
| Flappy | 0 | **1** | 0 | 0 |
| BallBeam | 1 | 0 | 0 | 0 |
| Catch | 0 | 0 | 0 | **1** |

Accuracy is 3/4. With only four trials this is suggestive rather than
conclusive: one-sided exact binomial probability of at least 3/4 successes at
a 25% chance rate is `p = 0.051`.

## Per-run notes

### CartPole — correct, 82%

The model paired `tensor_06 [256,4]` with `tensor_04 [256]`,
`tensor_07 [192,256]` with `tensor_03 [192]`, `tensor_05 [2,192]` with
`tensor_01 [2]`, and `tensor_02 [1,192]` with `tensor_00 [1]`. It identified
the policy head as almost antisymmetric (`tensor_01 = [-0.0312468, 0.0312468]`)
and the trunk as dominated by the third observation, coupled with the fourth:
features resembling `theta + k * theta_dot`. That is the expected CartPole
balancing signature. It explicitly considered BallBeam but rejected it because
it expected BallBeam's ball-position/velocity pair to be more prominent.

### Flappy — correct, 87%

The model noticed that the first and third input columns of `tensor_06` are
systematically opposed, producing features like `obs2 - obs0`. It read this as
`gap_y - bird_y`, i.e. the relative vertical error to the next gap. It also
noted the near-antisymmetric policy head and interpreted the action as
flap/glide. Both the observation and action interpretations match the Flappy
environment.

### BallBeam — incorrect, predicted CartPole at 88%

The model again found a policy driven mainly by the third observation and
coupled to the fourth. It interpreted this as a pole-angle/angular-velocity
controller and wrote:

> "The learned policy therefore behaves like a left/right balancing controller
> whose dominant decision variable is the pole angle ... That is the CartPole
> behavior rather than ... a BallBeam full-state beam controller."

In BallBeam the third observation is the **beam angle**, not a pole angle. The
same state ordering and the same kind of balancing controller make the two
checkpoints genuinely hard to separate from weights alone. The model's
rejection of BallBeam rested on the prior that ball position/velocity should
dominate; this particular BallBeam policy is angle-dominated instead.

The answer also described the hidden activation as ReLU, while the trained
network uses `tanh` (see `rl/model.py`). This does not affect the tensor
pairing, but it is an incorrect architectural detail.

### Catch — correct, 88%

The model found a strong opposition between the first two input columns, i.e.
features proportional to `obs0 - obs1`, and read it as `paddle_x - ball_x`. It
also noted that the fourth input column is comparatively weak and that the
policy head is symmetric/odd, both consistent with a paddle-tracking policy.
One minor semantic slip: it described the weak fourth observation as a "fixed
vertical speed", but in this environment the fourth observation is the ball's
horizontal velocity; the third is the ball's vertical position. The core
relative-horizontal-position argument is still correct.

## Token usage and reasoning

- Total billed tokens: **1,578,771**.
- Prompt tokens: **1,308,427** (82.9% of total).
- Completion tokens: **270,344**, of which **266,965** were reasoning tokens
  (98.8% of completion).
- Average per run: **394,693** total tokens.
- The BallBeam run consumed the most output (119,986 completion tokens,
  119,330 reasoning tokens) and still produced the only wrong answer. Long
  reasoning did not protect against the CartPole/BallBeam confusion.

Self-reported confidence was high in every run (82–88%) and did not
discriminate correct from incorrect answers: the BallBeam error was tied for
the highest confidence in the set.

## Observations

1. **Tensor pairing was reliable.** Every run correctly reconstructed the
   actor-critic data flow and paired each weight matrix with its bias.
2. **Distinct dynamics were recognised.** Flappy's relative vertical error and
   Catch's relative horizontal error were both identified from the first-layer
   column structure. CartPole's angle/angular-velocity coupling was also found.
3. **The failure was an isomorphic-task ambiguity.** CartPole and BallBeam
   share the observation schema `[position, velocity, angle, angular velocity]`
   and symmetric two-action control. Weight-space evidence for "balance an
   angle" does not, by itself, identify which physical system produced it.
4. **Confidence was not calibrated in this sample.** The model was highly
   confident on all four answers, including the single error.
5. **Reasoning was extremely long.** The four runs produced 163k–340k
   characters of reasoning each and 46.8k–119.3k reasoning tokens. The hardest
   run (BallBeam) spent roughly 2.5x the reasoning tokens of CartPole or
   Flappy.

## Limitations

- One sample per checkpoint and one model, so there is no estimate of
  run-to-run variance. With `n = 4`, the 3/4 result has wide uncertainty.
- The runs did not use `--description-file`, although the harness documents it
  as the recommended neutral release description for policy checkpoints. The
  prompt still contained the four candidates and the architecture facts.
- No retries or alternative prompts were tested. The observed CartPole/BallBeam
  confusion may be stable or may vary with prompt wording; this experiment
  cannot distinguish those cases.
- No cost estimate is reported because the run records do not include pricing.

## Reproducing

Run from `wsp-read-harness/` with a key supplied through the environment
(the literal key from the original command is intentionally omitted):

```bash
python wsp_read.py \
  --base-url https://api.deepseek.com \
  --api-key "$DEEPSEEK_API_KEY" \
  --model-name deepseek-flash \
  --model ../anonymous/policy_ballbeam.safetensors \
  --thinking enabled \
  --reasoning-effort max \
  --save ../outputs/dsv4.1-flash-1017-ballbeam.json
```

Use `policy.safetensors`, `policy_flappy.safetensors`,
`policy_ballbeam.safetensors` and `policy_catch.safetensors` for CartPole,
Flappy, BallBeam and Catch respectively.

## Bottom line

DeepSeek V4.1 Flash 1017 is good at reading this actor-critic's weights: it
recovered the architecture, the policy symmetry, and the dominant observation
features in all four runs, and named three tasks correctly. Its one failure is
the CartPole/BallBeam pair, whose state layouts and control semantics are
nearly isomorphic. For this benchmark, 3/4 is a meaningful signal that
weight-reading works, while the BallBeam error shows that some task pairs need
more than static weight inspection to disambiguate.

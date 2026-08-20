# Reinforcing the World's Edge

Code and experiment statistics for:

> **Reinforcing the World's Edge: A Continual Learning Problem in the Multi-Agent-World Boundary**  
> Dane Malenfant, 2026. [arXiv:2603.06813](https://arxiv.org/abs/2603.06813)

The paper studies ordinary decentralized multi-agent reinforcement learning from the perspective of one focal agent. Although the joint Markov game is stationary, learning peers generate an episode-indexed sequence of induced MDPs for the focal agent. The invariant core tracks success-conditioned reusable structure in that continual process, and the theory quantifies its lifetime under peer-policy drift.

## Repository contents

- `code/*.py`: synthetic experiments, structural-law tests, and figure-regeneration scripts.
- `code/continual_control_toy_v1_confirmation_stats/`: registered continual-control development and 64-stream confirmation results.
- `code/continual_cue_mnist_v4_confirmation_stats/`: registered 64-stream cue-MNIST confirmation results.
- `code/lbf_v3_core_erosion_exploratory_stats/`: exploratory learned-partner LBF erosion analysis.

## Regenerate the figures

The figure code requires Python 3, NumPy, and Matplotlib.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install numpy matplotlib
python3 code/regenerate_all_figs.py
```

The complete regeneration pass writes vector figures and numerical summaries to `figs/` and takes approximately 3--5 minutes on one CPU core.

Individual confirmation figures can also be regenerated directly:

```bash
python3 code/plot_continual_control_confirmation.py
python3 code/plot_cue_mnist_confirmation.py
python3 code/plot_lbf_core_erosion_exploratory.py
```

## Citation

```bibtex
@misc{malenfant2026reinforcingworldsedgecontinual,
      title={Reinforcing the World's Edge: A Continual Learning Problem in the Multi-Agent-World Boundary}, 
      author={Dane Malenfant},
      year={2026},
      eprint={2603.06813},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2603.06813}, 
}
```

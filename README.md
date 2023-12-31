# Amortizing intractable inference in large language models

This repository contains code for GFlowNet fine-tuning of language models, as described in the paper

**Amortizing intractable inference in large language models**<br />
Edward J. Hu*, Moksh Jain*, Eric Elmoznino, Younesse Kaddar, Guillaume Lajoie, Yoshua Bengio, Nikolay Malkin <br/>
Paper: https://arxiv.org/abs/2310.04363
<details>
<summary>
BibTeX
</summary>
  
```bibtex
@article{hu2023amortizing,
  title={Amortizing intractable inference in large language models},
  author={Hu, Edward J. and Jain, Moksh and Elmoznino, Eric and Kaddar, Younesse and Lajoie, Guillaume and Bengio, Yoshua and Malkin, Nikolay},
  year={2023},
  journal={arXiv preprint 2310.04363}
}
```
</details>

Visit the subdirectories to find code and documentation for each experiment in the paper:
- Random number generation (§2): `rng`
- Sentence continuation (§4.1): `next_sentence`
- Story infilling (§4.2): `infill_subj_arithmetic`
- Subjectivity classification (§4.3): `infill_subj_arithmetic`
- Arithmetic with tool use (§4.4): `infill_subj_arithmetic`

Please contact us or post an issue if you have any questions.

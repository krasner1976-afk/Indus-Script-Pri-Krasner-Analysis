# Quantitative Evaluation of the Indus Script Positional Entropy

This repository contains the computational epigraphy pipeline used to evaluate the **Commodity-Accounting Paradigm** of the Indus Valley script and verify the **Pri-Krasner Transaction-Limit Hypothesis**.

## Repository Structure
- `indus-entropy-analysis.py` - Core Python 3 script containing entropy calculations, simulated database construction, and statistical validation.
- `README.md` - Reproduction guide and documentation.

## Theoretical Background
We analyze the positional Shannon entropy of 5-sign Harappan inscriptions. Computational complexity models identify a statistically significant information "dip" in the penultimate position (Slot 4) of approximately **3.77 bits**, compared to the highly randomized merchant ID slots (Slots 2 & 3, ~4.44 bits). This repository provides a reproducible framework proving that this structural constraint is mathematically consistent with an administrative/transactional ledger system.

## Requirements
To run this analysis locally, you need Python 3.10+ and the following scientific libraries:
```bash
pip install pandas numpy scipy matplotlib

# Motor Imagery BCI for Stroke Rehabilitation — Python rewrite

Python/scikit-learn rewrite of a MATLAB motor-imagery BCI pipeline from the
IEEE SMC BR41N.IO Hackathon (g.tec Medical Engineering), decoding imagined
left vs. right hand movement from 16-channel EEG for stroke rehabilitation.
3 patients, each with a pre- and post-therapy session (25 BCI sessions of
therapy in between), 80 trials/class per session.

## What changed from the original MATLAB version

- **Bandpass narrowed to 8–30 Hz** (mu+beta), not the original 1–40 Hz
  broadband — matches the ERD/ERS-relevant band used in the published
  methodology this repo replicates.
- **CSP is now regularized** (`reg="ledoit_wolf"`). Common average
  referencing makes the channel covariance matrix rank-deficient by
  construction; without shrinkage, CSP's eigenvalue solve can fail outright
  on real data, not just underperform.
- **The original filterbank-CSP attempt (`new_main.m`) had a bug**: it
  built a proper 9-band filterbank but then sliced to `(:,:,1)` — band 1
  is 4–8 Hz (theta), so the CSP transform never actually saw mu or beta.
  This repo's CSP+LDA baseline fixes that by operating directly on a
  correctly-banded 8–30 Hz signal instead.
- **Riemannian geometry classifiers added** (`pyriemann`): tangent-space
  logistic regression and MDM. These are the current standard for
  small-trial-count MI-BCI and don't require the component-count tuning
  CSP does.

## Benchmarks from the original hackathon (BCI 2023-Winter School)

| Subject | CSP+LDA | PCA+TVLDA |
|---|---|---|
| P1 pre  | 79.7% | 100.0% |
| P1 post | 68.4% | 72.4% |
| P2 pre  | 77.1% | 92.9% |
| P2 post | 93.9% | 97.0% |
| P3 pre  | 96.1% | 97.4% |
| P3 post | 74.4% | 93.6% |

PCA+TVLDA (Gruenwald et al., 2019) beat standard CSP+LDA by +10.6 points on
average. This repo's CSP+LDA baseline is meant to at least match the
published CSP+LDA column; the Riemannian pipelines are the candidate for
closing some of that gap without reimplementing TVLDA itself.

## Setup

```bash
pip install -r requirements.txt
```

Place your `.mat` session files in `data/` (git-ignored), using the
original naming convention:

```
data/
  P1_pre_training.mat   P1_pre_test.mat
  P1_post_training.mat  P1_post_test.mat
  P2_pre_training.mat   P2_pre_test.mat
  ...
```

Each file must contain `y` (samples × channels), `trig` (+1 left, -1
right, 0 rest), and `fs` (sampling rate), per the original README.

## Run

```bash
python scripts/01_run_all_models.py
```

Runs CSP+LDA, Riemannian tangent-space+LR, and Riemannian MDM across every
subject × phase, saves `results/model_comparison.csv`, and prints each
result alongside the published CSP+LDA / PCA+TVLDA benchmarks above.

## Repo structure

```
config.py              # paths, channel montage, band, trial window
src/
  data_loading.py       # .mat file loading
  preprocessing.py      # CAR, 8-30Hz bandpass, trial extraction
  modeling.py            # CSP+LDA, Riemannian TS+LR, Riemannian MDM
  evaluation.py           # per-subject/phase runner, metrics (ported from RFsimple.m/SVMlinear.m)
scripts/
  01_run_all_models.py
data/                   # your .mat files (git-ignored, never committed)
results/                # model_comparison.csv
figures/                # (reserved for later showcase plots)
```

## References

- Gruenwald et al. (2019). Time-Variant Linear Discriminant Analysis
  Improves Hand Gesture and Finger Movement Decoding for Invasive
  Brain-Computer Interfaces. *Frontiers in Neuroscience*.
- Sebastián-Romagosa et al. Brain Computer Interface treatment for motor
  rehabilitation of upper extremity of stroke patients — A feasibility
  study. *Frontiers in Neuroscience*.

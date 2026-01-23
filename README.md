# Multi-Case CFD Data Analysis

ML analysis of OpenFOAM simulations - flow over wall-mounted cube with different inlet profiles.

## Structure
```
├── data/
│   ├── raw/           # OpenFOAM cases (not tracked)
│   └── processed/     # Extracted profiles (not tracked)
├── notebooks/         # Analysis notebooks
├── src/               # Reusable code
└── results/           # Figures and reports
```

## Setup
# locally on the PC:
condaON                   # alias condaON='source ~/anaconda3/bin/activate

# globally on any device:
```bash
conda env create -f environment.yml
conda activate ml_cfd
```

## Workflow
Step 0
- get together the data from CFD simulations: ../Data
- run Postpro to generate centre points file C in latestTime directory: ../Data/generate_cell_centers.sh

Step 1
- Load all data: ./Step1__Load_all_cases.ipynb
    = Extract profiles from OpenFOAM

Step 2
- Data analysis: ./Step2__Data_analysis.ipynb
    = PCA, t-SNE, statistical analysis

Step 3


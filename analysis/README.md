The contents of this folder include 3 notebooks used to analyze and visualize the raw data available at: https://doi.org/10.5061/dryad.bnzs7h4c4.

As described in each notebook, you will need to download the data first.

# Notebooks

### Calibration_elasticity-sweeps.ipynb
For each scenario, we perform a grid search over elasticity parameters to find which parameter value yields the best social welfare under the Saez tax formula.

The grid search is used to calibrate subsequent experiments. Specifically, we set the elasticity parameter based on its best-performing value. In some scenarios, different values are optimal for different definitions of social welfare.

This notebook visualizes the results of the grid search for each of the 6 GTB scenarios.

In addition, this notebook visualizes the results of a sweep over fixed tax rates in order to empirically quantify elasticity. This visualization replicates a paper figure, which includes a comparison to the results of the elasticity grid search.

### Process-dense-logs.ipynb

The main analysis/visualization notebook (see below) uses preprocessed data based on raw episode dense logs. This notebook provides a reference for how this preprocessing is perfored and allows for other sets of experiments to be preprocessed, if desired. New sets of preprocessed data could, for example, be used to extend the analyses presented in the main notebook.

### Experiment-analysis-and-visualization.ipynb

This is the main analysis/visualization notebook. The purpose of this notebook is to reproduce all the figures included in our paper (except for the one figure covered in the calibration notebook).

The data used in this notebook comes both from training history data (included in the raw data that must be downloaded) and preprocessed data, which is included in this directory in

```preprocessed_dense_logs_stats_dict-open_quadrant_4-eq_times_prod-used_in_paper_analyses.pkl```

In addition, the tax gaming figure uses an example dense log which is included in this directory as well (as ```tax_gaming_reference_dense_log-ai_economist.json```).

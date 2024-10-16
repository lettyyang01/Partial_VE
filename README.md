# Algebraic and Statistical Properties of the Partially Regularized Ordinary Least Squares Interpolator

This repository contains the code for the paper *Algebraic and Statistical Properties of the Partially Regularized Ordinary Least Squares Interpolator*. The simulation settings are detailed in the Section 6 in the article.

## Usage

To replicate Figures 1-4, 

1. Start by running `simu.sh` to generate the tables for the estimates.

Note: If you want to run a specific simulation, modify the `EXPERI_TYPE_LIST` variable in the `.sh` file. You can also adjust the `VE_TYPE_LIST` and `COVAR_TYPE_LIST` variables to focus on particular variance estimators or generative models.

2. Once the estimate tables are generated as `.csv` files, use the `draw_VE_plot.ipynb` notebook to create the plots. Detailed instructions are provided inside the notebook.

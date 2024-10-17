# Algebraic and Statistical Properties of the Partially Regularized Ordinary Least Squares Interpolator

This repository contains the code for the paper *Algebraic and Statistical Properties of the Partially Regularized Ordinary Least Squares Interpolator*. 

## Usage

We recommend using a high-performance computing cluster to run the simulations, as they require intensive computation. Additionally, the simulations rely on "ray" for parallel execution.

### Reproduce the Motivation Example in Section 1

See the simulation settings in the Appendix. To replicate **Figure 1**:

1. Run `begin_eg.sh`.
2. Open and run the notebook `resultProcessing/ate_bias.ipynb` to visualize the results.  
   **Note:** You may need to modify the directory path where the results are saved.

---

### Reproduce the Results in Section 5

See the simulation settings in the main text. To replicate **Figures 2-5**:

1. Run `simu.sh` to generate the estimation tables.  

   **Note:**  
   - If you wish to run a specific simulation, modify the `EXPERI_TYPE_LIST` variable in the `.sh` file.  
   - You can also adjust the `VE_TYPE_LIST` and `COVAR_TYPE_LIST` variables to focus on specific variance estimators or generative models.

2. Once the tables are generated as `.csv` files, open and run the `resultProcessing/draw_VE_plot.ipynb` notebook to generate the plots. Detailed instructions are provided within the notebook.
   
   **Note:** You may need to modify the directory path where the results are saved.

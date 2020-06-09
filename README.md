# PMIME-and-TE
We realize partial conditional mutual information from mixed embedding for coupling estimation in multivariate time series. We also use transfer entropy to realize this method.
In the proposed measure of Partial MIME (PMIME), the embedding is on all observed variables, and it is optimized in explaining the response variable. It is shown that PMIME detects correctly direct coupling, and outperforms the (linear) conditional Granger causality and the partial transfer entropy. And we also use transfer entropy instead of conditional mutual information to evaluate the coupling.
# Environment
- matlab
- windows or linux or mac
# Step
1. prepare the data which means the N x K matrix of the K time series of length N.

2. Coding the program which uses PMIME or TE.

```
    function [RM,ecC] = PMIME(allM,Lmax,T,nnei,A,showtxt)
    function [RM,ecC] = TransferEntropy(allM,Lmax,T,nnei,nsur,alpha,showtxt)
```
## INPUT
- **allM:** the N x K matrix of the K time series of length N.

- **Lmax:** the maximum delay to search for X and Y components for the mixed embedding vector [default is 5].

- **T:** T steps ahead that the mixed embedding vector has to explain. Note that if T>1 the future vector is of length T and contains the samples at times t+1,..,t+T [dafault is 1].

- **nnei:** number of nearest neighbors for density estimation [default is 5].

- **A:** the threshold for the ratio of CMI over MI of the lagged variables for the termination criterion.

- **showtxt:** if 0 or negative do not print out anything, if 1 print out the response variable index at each run, if 2 or larger print also info for each embedding cycle [default is 1].

- **nsur:** the number of surrogates for the significance test [default is 100].

- **alpha:** the significance level for the test for the termination criterion [default is 0.05].

## OUTPUT
- **RM:** A K x K matrix containing the R values computed by PMIME using surrogates for setting the stopping criterion.

- **ecC:** cell array of K components, where each component is a matrix of size E x 5, and E is the number of embedding cycles. For each embedding cycle the following 5 results are stored:
   1. variable index
   2. lag index
   3. CMI of the selected lagged variable w and the future response state x^F given the current mixed embedding vector,I(x^F; w| wemb)
   4. MI between x^F and the augmented mixed embedding vector [wemb w], I(x^F; w,wemb). 
   5. The ration of 3 and 4: I(x^F; w| wemb)/I(x^F; w,wemb)

# Licence
- the GNU General Public License

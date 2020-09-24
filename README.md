# Bachelor-thesis
julia-code for bachelor thesis written in Version 1.5

The get_ready.jl file will install all the nessesery julia packages.

The .xlsx file consists of the reported COVID-19 cases according to the RKI (19th of July 2020) and the data transformation.

The .csv file is for easier handling in julia.


The Bachelorabreit.jl file:

1. RKI-ODE-model
2. Parameter and initial condition sensitivity (parametersample, initsample, odesample, DataQuantile, sensitivityplot)
3. SEIR-model and cumulative case numbers (RKI_COVID19.csv)
4. Random Walk Metroplois Algorithm for adjusted SEIR-parameters (myRWM, θtransform, πp; Autocorrelation, Traceplot, Sample distribution, Correlation)

# mlhc_code

Code for MLHC2020 paper on glucose prediction using OpenAPS and OhioT1DM data

Usage:
In the terminal run the following commands twice. First with `dataset='oaps'` and then with `dataset='ohio'` in `preprocess.sh` and `run.sh` files:  

`chmod +x ./preprocess.sh`  
`./preprocess.sh`   

`chmod +x ./run.sh`  
`./run.sh`

In run.sh file, set the following parameters according to the experiment you
are running.  
`dataset='ohio' (ohio or oaps)`  
`history_window = 12 (number of past glucose values to use. 12 samples means an hour of previous data (frequency = 5 minutes) `   
`prediction_window = 60 (30 minutes or 60 minutes)`  
`dimension = multivariate (univariate or multivariate)`  
`prediction_type = single (single or multi. This refers to single step or multioutput forecasting)`  
`normalize_data = False (True or False. use normalized data or not)`  
`model_name = RNN (['REG','SVR','TREE','ENSEMBLE','LSTM', 'RNN'])`  
`save_results = False (True or False. It will replace old output files in the output directory)`  
`ablation_code = 0 (0 = unfiltered_imputed, 1.filtered_imputed, 2.unfiltered_unimputed 3. filtered_unimputed)  `
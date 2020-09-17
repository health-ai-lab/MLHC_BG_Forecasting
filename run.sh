#!/bin/sh
dataset='oaps' #ohio or oaps
if [ "$dataset" = "ohio" ]; then
    root_directory="../../../../PHI/PHI_OHIO/" 
    data_directory=$root_directory"data/csv_files/" 
fi
if [ "$dataset" = "oaps" ]; then
    root_directory="../../../../PHI/PHI_OAPS/" 
    data_directory=$root_directory"OpenAPS_data/n=88_OpenAPSDataAugust242018/" #../../../data/PHI/PHI_OAPS/OpenAPS_data/n=88_OpenAPSDataAugust242018/

fi
output_directory=$root_directory"sandbox/hhameed/"
history_window=12 
prediction_window=30
dimension=univariate
prediction_type=single #single-step or multi-output
normalize_data=False
model_name=LSTM #['REG','SVR','TREE','ARIMA','LSTM', 'RNN']
save_results=True
ablation_code=1 #0.unfiltered_imputed, 1.filtered_imputed, 2.unfiltered_unimputed 3. filtered_unimputed
if [ "$model_name" = "ARIMA" ]; then
    python $PWD/arima.py $root_directory $data_directory $output_directory $normalize_data $model_name $dataset $save_results $ablation_code
else
    python $PWD/main.py $root_directory $data_directory $output_directory $history_window $prediction_window $dimension $prediction_type $normalize_data $model_name $dataset $save_results $ablation_code
fi

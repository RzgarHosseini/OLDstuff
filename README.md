# CCSynergy
In the codes folder, you can find four versions of the CCSynergy program that includes codes corresponding to both CV1 and CV2 cross validation schemes and  the regression  and classification based settings.

## Inputs

### Cell line representation files
We have used five different approaches for representing cell lines, which are summarized as .csv files in the Data folder (e.g. Data/Cell1.csv).
Each column in the csv files corresponds to a given cell lines, which is represented as a low-dimentional vector of length 100.

### Drug synergy datasets
We have used two distinct datasets, the Merck dataset (Data/Merck.csv) and the Sanger dataset (Data/Sanger.csv), which we have used respectively for predictions in a regression (Loewe synergy scores) or classification (Binary scores) based setting.  

### Drug Features (Chemical Checker extended drug similarity profiles)
The Metrics sub-folder includes the 25 CC feature spaces for all single drugs. The .csv files in the Data/Metrics/Merck sub-folder contain 36 rows each of which corresponds to a given drug specified in the Data/Metrics/Drugs_Single_Info_Merck.csv. Similarly, the .csv files in the Data/Metrics/Sanger sub-folder contain 62 rows each of which corresponds to a given drug specified in the Data/Metrics/Drugs_Single_Info_Sanger.csv.

### Parameters
The code requires the following parameters to be passed by the user.

We have fixed the following optimized hyper-parameters:

## Examples:

## Output

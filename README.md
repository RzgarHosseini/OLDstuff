# CCSynergy
In the codes folder, you can find four versions of the CCSynergy program that includes codes corresponding to both CV1 and CV2 cross validation schemes and  the regression  and classification based settings.

## Inputs

### Cell line representation files
We have used five different approaches for representing cell lines, which are summarized as .csv files in the Data folder (e.g. Data/Cell1.csv).
Each column in the csv files corresponds to a given cell lines, which is represented as a low-dimentional vector of length 100.

### Drug synergy datasets
We have used two distinct datasets, the Merck dataset (Data/Merck.csv) and the Sanger dataset (Data/Sanger.csv), which we have used respectively for predictions in a regression (Loewe synergy scores) or classification (Binary scores) based setting.  

### Drug Features (Chemical Checker extended drug similarity profiles)
The Metrics sub-folder includes the 25 CC feature spaces for all single drugs (36 single drugs in Merck dataset: Data/Metrics/Merck and the 62 single drugs in the Sanger dataset: Data/Metrics/Sanger). 

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


DIR: The full path to the working directory <br>
j:   Index for cell line representation method (1-5) <br>
i:   Index for the 25 drug representation CC spaces (1-25: corresponding to A1-E5) <br> 
TINDX:  Index for a given tissue, which varies between 1 and 6 (in the merck dataset) and between 1 and 3 (in the Sanger dataset). You can see the mapping tissues to their corresponding indices from the "Tissue" and "Tindex" columns in the Data/Merck.csv and Data/Sanger.csv files. Note that this parameter is necessary only in the CV2 codes.
    
We have fixed the following optimized hyper-parameters:

n1=2000   (number of neurons in the first layer) <br>
n2=1000   (number of neurons in the second layer) <br>
n3=500    (number of neurons in the third layer) <br>
lr=0.0001 (learning rate) <br>
batch=128 (batch size) <br>
seedd=94  (seed number) <br> 
num=356   (size of the input vector) <br>

## Examples:

## Output

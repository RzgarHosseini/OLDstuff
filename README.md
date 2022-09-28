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


## Output
The final output file is a .csv file that specifies the real and predicted value of the (drug pair + cell line) triplets, for each of which the fold index and their corresponding index in the original dataset are specified. This output file is then used for calculating the relevant evaluation metrics. Furthermore, the trained model is saved in a seperate directory.

## Examples:
### python3 /Codes/CV1_Regression.py /Home/User/CCSynergy 3 12 <br> 
In this example, drug synergy is predicted using the Merck dataset under CV1 scheme. The CCSynergy method III (i.e.e CARNIVAL-based signaling pathway activity scores) is used for cell line representation and C2 CC signatures (12-th CC space) are used for encoding the drug features. <br> 

The output will be saved in /Home/User/CCSynergy/DNN_CV1_Regression_Cell3_C2.csv file and the information regarding the trained model will be stored in /Home/User/CCSynergy/DNN_CV1_Regression_Cell3_C2 directory. <br>   

### python3 /Codes/CV2_Classification.py /Home/User/CCSynergy 5 18 2 <br> 
In this example, drug synergy is predicted using the Sanger dataset under CV1 scheme. The CCSynergy method V (i.e.e DepMap-based signaling pathway dependency scores) is used for cell line representation and D3 CC signatures (18-th CC space) are used for encoding the drug features. In this example, the the index for testing tissue is 2, which corresponds to "breast" in the Sanger dataset. <br>  
The output will be saved in /Home/User/CCSynergy/DNN_CV2_Classification_Cell5_D3_2.csv file and the information regarding the trained model will be stored in /Home/User/CCSynergy/DNN_CV2_Classification_Cell5_D3_2 directory. <br> 

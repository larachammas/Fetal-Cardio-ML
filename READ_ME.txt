

For TESTING THE FINAL MODELS only run: 
	- finalMLP.m 
	- finalSVM.m



Other Files and workflow:

1. data.csv (initial data from UCI used in data_processing.m)

2. ADASYN.m (Function from https://www.mathworks.com/matlabcentral/fileexchange/50541-adasyn-improves-class-balance-extension-of-smote, used in data_processing.m)

3. data_processing.m (for data cleaning and preprocessing)

4. finalized_Data.csv (the cleaned data created and saved by the data_processing.m file)

5. train_test_split.m (for splitting the data into training and testing. Saves the training and testing data in the 'data.mat' file)

6. data.mat (training and testing data needed split into classes and features) 

7. trainingMLP.m (grid search and final MLP model, takes about 40 minutes)

5. trainingSVM.m (grid search and final SVM model, takes about 3 hours)

**Meant to be run in Matlab R2018b**

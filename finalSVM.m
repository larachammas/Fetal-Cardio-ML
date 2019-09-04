%% Running Final Models and Printing Metrics %% 
clear all ; clc ; close all 

load('data.mat') %retrieve the data to use

%Convert the class labels to categorical variables
classes_Testing = categorical(Testing(:,25));
classes_Training = categorical(Training(:,25));

%% Create the final model 
finalModel = fitcecoc(features_Training, classes_Training, 'Coding', 'onevsone', ...
           'Learners', templateSVM('KernelFunction', 'polynomial', 'PolynomialOrder', 3, ...
           'BoxConstraint', 0.4), 'Verbose', 1);

%Predict on unseen testing data  
[test_labels,test_scores] = predict(finalModel, features_Testing);
 
%% Generate Evaluation Metrics for final model on testing data

testingConfusion= confusionmat(classes_Testing,test_labels)%calculate confusion matrix
finalAccuracy = 100*sum(diag(testingConfusion))./sum(testingConfusion(:)) %calculate accuracy

%Class 1 precision, recall and F1 score calculation
testing_precision_class1 = testingConfusion(1,1) / ...
    (testingConfusion(1,1) + (testingConfusion(2,1)+testingConfusion(3,1)));

testing_recall_class1 = testingConfusion(1,1) / ...
    (testingConfusion(1,1) + (testingConfusion(1,2)+testingConfusion(1,3)));

testing_F1Score_class1 = 2 * ((testing_precision_class1*testing_recall_class1)/ ...
    (testing_precision_class1+testing_recall_class1));

% Class 2 precision, recall and F1 score calculation
testing_precision_class2 = testingConfusion(2,2) / ...
    (testingConfusion(2,2) + (testingConfusion(1,2)+testingConfusion(3,2)));

testing_recall_class2 = testingConfusion(2,2) / ...
    (testingConfusion(2,2) + (testingConfusion(2,1)+testingConfusion(2,3)));

testing_F1Score_class2 = 2 * ((testing_precision_class2*testing_recall_class2)/...
    (testing_precision_class2+testing_recall_class2));

% Class 3 precision, recall and F1 score calculation
testing_precision_class3 = testingConfusion(3,3) / ...
    (testingConfusion(3,3) + (testingConfusion(1,3)+testingConfusion(2,3)));

testing_recall_class3 = testingConfusion(3,3) / ...
    (testingConfusion(3,3) + (testingConfusion(3,1)+testingConfusion(3,2)));

testing_F1Score_class3 = 2 * ((testing_precision_class3*testing_recall_class3)/...
    (testing_precision_class3+testing_recall_class3));

%Final Table combining all the classes results 
testingMetrics = [testing_precision_class1, testing_recall_class1, testing_F1Score_class1; testing_precision_class2, ...
    testing_recall_class2, testing_F1Score_class2; testing_precision_class3, testing_recall_class3, testing_F1Score_class3];
testingMetrics = array2table(testingMetrics);
testingMetrics.Properties.VariableNames = {'Precision', 'Recall', 'F1'}

%Print confusion matrix on testing data 
figure(1)
confusionchart(classes_Testing,test_labels)
title('SVM Confusion Matrix of Testing Data')
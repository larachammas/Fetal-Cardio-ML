clear all ; clc ; close all 

%Load in the data environment which has them pre-split 
load('data.mat')

%Convert the class labels to categorical variables
classes_Testing = categorical(Testing(:,25));
classes_Training = categorical(Training(:,25));

%Set hyperparameters to test 
codingDesign = ["onevsall"; "onevsone"];
svmKernel = ["gaussian"; "linear"; "polynomial"];
box = [0.2, 0.3, 0.4]; %values baesd on work from ocak 2013
polynomialOrder = [2,3,4];

%Initalize final tables which will store model training results
Accuracy = [];
FinalAll = {'codingDesign', 'box size', 'Kernel', 'polynomialOrder','Accuracy'};


%% Grid search testing every combination of hyperparameter possible 
%using 10-fold cross-validation to train and tune each model

for j = 1:length(svmKernel) %iterate through kernal types
    if svmKernel(j) ~= 'polynomial' %if the kernal is NOT polynomial (ie guassian or linear), continue
        
         for i = 1: length(codingDesign) %iterate through coding designs
             
            for g = 1:length(box) %iterate through box constraints 
                
                %specify SVM learner parameters based on value selected in each loop 
                
                learners =  templateSVM('KernelFunction', svmKernel(j), 'BoxConstraint', box(g)) ; 

                %train model using these parameters with 10-fold
                %crossvalidation
                
                rng(19) %set a seed for reproducability 
                Mdl = fitcecoc(features_Training, classes_Training, 'Coding',  ...
                codingDesign(i),'Learners', learners, 'Crossval', 'on', 'Verbose', 1); 
            
                %predict labels of the left-out data from each cross-validiation fold 
                [tuning_labels,tuning_scores]= kfoldPredict(Mdl);
                
                %generate confusion matrix on these predictions
                CM_model= confusionmat(classes_Training,tuning_labels); 

                %calculate the accuracy of the model using the confusion matrix 
                Accuracy_tuning = 100*sum(diag(CM_model))./sum(CM_model(:));
       
                %Store the parameters tested and accuracy of corresponding model together
                Accuracy = [Accuracy ; Accuracy_tuning]; 
                FinalAll = [FinalAll ; {codingDesign(i), box(g), svmKernel(j), ' ' ,Accuracy_tuning}];
            end 
         end 
        
%if the kernal is polynomial (ie everything else in the list)
%continue with the same steps 
%however adding polynomial order as a hyperparameter to the learner             
    else 
        
        for k = 1:length(polynomialOrder)
            
            for i = 1: length(codingDesign)
                
                for g = 1:length(box)
                     %specify SVM learner parameters based on value selected in
                     %each for loop
                    learners =  templateSVM('KernelFunction', svmKernel(j),'PolynomialOrder', polynomialOrder(k),...
                        'BoxConstraint', box(g)) ; 
                    
                    %train model
                    rng(19) %set a seed for reproducability 
                    Mdl = fitcecoc(features_Training, classes_Training, 'Coding',  ...
                    codingDesign(i),'Learners', learners, 'Crossval', 'on', 'Verbose', 1); 

                    %predict labels 
                    [tuning_labels,tuning_scores]= kfoldPredict(Mdl);

                    %generate confusion matrix
                    CM_model= confusionmat(classes_Training,tuning_labels); 

                    %calculate the accuracy of the model using the confusion matrix 
                    Accuracy_tuning = 100*sum(diag(CM_model))./sum(CM_model(:));

                    %Store everything together
                    Accuracy = [Accuracy ; Accuracy_tuning]; 
                    FinalAll = [FinalAll ; codingDesign(i), box(g), svmKernel(j), polynomialOrder(k), Accuracy_tuning];
                end 
            end 
        end 
    end
end

%% Selecting the best hyperparameters 

%assign column names to table
FinalAll = array2table(FinalAll);
FinalAll.Properties.VariableNames = {'codingDesign', 'BoxContraint','Kernals', 'Polynomial', 'Accuracy'} ; 

%find the highest model accuracy in accuracy table
highestAccuracy = max(Accuracy(:,1))
highestAccuracy = num2str(highestAccuracy)

%find the model row with the highest accuracy and what its parameters are 
best_model = FinalAll(FinalAll.Accuracy == highestAccuracy, :)

%Store the optimal parameters into variables for later use 
bestCodingDesign = best_model{:,1} ; 
bestBox = str2num(best_model{:,2});
bestLearner = best_model{:,3};
bestP = str2num(best_model{:,4});

%% Train and test new model using best hyperparameters
       
finalModel = fitcecoc(features_Training,classes_Training, 'Coding', bestCodingDesign, ...
           'Learners', templateSVM('KernelFunction', bestLearner, ...
           'BoxConstraint', bestBox), 'Verbose', 1);
       
%Predict on previously seen training data 
[training_labels,training_scores] = predict(finalModel, features_Training);

%Predict on unseen testing data  
[test_labels,test_scores] = predict(finalModel, features_Testing);
 
%% Generate Evaluation Metrics for final model on training data 

trainingConfusion= confusionmat(classes_Training,training_labels) %calculate confusion matrix
trainingAccuracy = 100*sum(diag(trainingConfusion))./sum(trainingConfusion(:)) %calculate accuracy

% Class 1 precision, recall and F1 score calculation
training_precision_class1 = trainingConfusion(1,1) / ...
    (trainingConfusion(1,1) + (trainingConfusion(2,1)+trainingConfusion(3,1)));

training_recall_class1 = trainingConfusion(1,1) / ...
    (trainingConfusion(1,1) + (trainingConfusion(1,2)+trainingConfusion(1,3)));

training_F1Score_class1 = 2 * ((training_precision_class1*training_recall_class1)/...
    (training_precision_class1+training_recall_class1));

%Class 2 precision, recall and F1 score calculation
training_precision_class2 = trainingConfusion(2,2) / ...
    (trainingConfusion(2,2) + (trainingConfusion(1,2)+trainingConfusion(3,2)));

training_recall_class2 = trainingConfusion(2,2) / ...
    (trainingConfusion(2,2) + (trainingConfusion(2,1)+trainingConfusion(2,3)));

training_F1Score_class2 = 2 * ((training_precision_class2*training_recall_class2)/...
    (training_precision_class2+training_recall_class2));

%Class 3 precision, recall and F1 score calculation
training_precision_class3 = trainingConfusion(3,3) / ...
    (trainingConfusion(3,3) + (trainingConfusion(1,3)+trainingConfusion(2,3)));

training_recall_class3 = trainingConfusion(3,3) / ...
    (trainingConfusion(3,3) + (trainingConfusion(3,1)+trainingConfusion(3,2)));

training_F1Score_class3 = 2 * ((training_precision_class3*training_recall_class3)/...
    (training_precision_class3+training_recall_class3));

%Final Table combining all the classes results 
trainingMetrics = [training_precision_class1, training_recall_class1, training_F1Score_class1; training_precision_class2, ...
    training_recall_class2, training_F1Score_class2; training_precision_class3, training_recall_class3, training_F1Score_class3];
trainingMetrics = array2table(trainingMetrics);
trainingMetrics.Properties.VariableNames = {'Precision', 'Recall', 'F1'}

%Print confusion matrix for training data 
figure(1)
confusionchart(classes_Training,training_labels)
title('SVM Confusion Matrix of Training Data')

%% Generate Evaluation Metrics for final model on testing data

finalConfusion= confusionmat(classes_Testing,test_labels)%calculate confusion matrix
finalAccuracy = 100*sum(diag(finalConfusion))./sum(finalConfusion(:)) %calculate accuracy

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
figure(2)
confusionchart(classes_Testing,test_labels)
title('SVM Confusion Matrix of Testing Data')

%% Generate the graphs for figure 2 showing how hyperparameter changes affect accuracy
FinalAll = FinalAll(2:31,:) %remove the first line which has the headers

%group box constraint and find the maximum value
groupsBox = findgroups(FinalAll(:,2));
Box_v_Accuracy = splitapply(@max, Accuracy(:,1), groupsBox);

%plot box constraint vs max accuracy
figure(3);
plot(Box_v_Accuracy)
xlabel('Box Constraint Value')
xticklabels({'0.2', '0.3', '0.4'})
ylabel('Accuracy (%)')
title('Box Constraint Vale vs SVM Accuracy')

%group coding design and find the maximum value
groupsCoding = findgroups(FinalAll(:,1));
coding_Accuracy = splitapply(@max, Accuracy(:,1), groupsCoding);

%plot coding design vs max accuracy
figure(4);
bar(coding_Accuracy)
xlabel('Coding Design')
xticklabels({'One vs All', 'One vs One'})
ylabel('Accuracy (%)')
title('Coding Design vs SVM Accuracy')

%group kernel type and find the maximum value
groupKernel = findgroups(FinalAll(:,3));
kernel_Accuracy = splitapply(@max, Accuracy(:,1), groupKernel);

%plot kernel type vs max accuracy
figure(5);
bar(kernel_Accuracy)
xlabel('SVM Kernel Type')
xticklabels({'Gaussian', 'Linear', 'Polynomial'})
ylabel('Accuracy (%)')
title('Kernel Type vs SVM Accuracy')

%group polynomial order and find the maximum value
groupPoly = findgroups(FinalAll(:,4));
poly_Accuracy = splitapply(@max, Accuracy(:,1), groupPoly);

%plot polynomial order vs max accuracy
figure(6); 
plot(poly_Accuracy)
xlabel('Polynomial Order')
xticklabels({'2', '3', '4'})
ylabel('Accuracy (%)')
title('Polynomial Order vs SVM Accuracy')

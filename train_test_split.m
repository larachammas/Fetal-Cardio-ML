%Import the data
data = csvread('finalized_Data.csv'); 

rng(10);

%shuffle the data
shuffledArray = data(randperm(size(data,1)),:);

%Creating a training and testing set 
[m,n] = size(shuffledArray) ;
P = 0.7 ;
idx = randperm(m)  ;
Training = shuffledArray(idx(1:round(P*m)),:) ; 
Testing = shuffledArray(idx(round(P*m)+1:end),:) ;

%Separate training data features and classes  
features_Training = Training(:,1:24);
classes_Training = Training(:,25);

%Separate testing data features and classes 
features_Testing = Testing(:,1:24);
classes_Testing = Testing(:,25);


save('data.mat')
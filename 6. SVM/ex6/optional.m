%% ==================== optional exercise: making my own dataset + training ===========================
% /data 에 있는 mail data 이용 중간 결과값들 words_data.mat에 저장해놓음 
% Part 1: make vocabList(2000 most frequent words in mails data)

clear; close all; clc;


words = {};
words_per_mail = [];

% extract words from emails and find most frequent words -> make vocabList

non_spam_data = dir('data/non_spam/*');
for i=3:length(non_spam_data)
    file_name = non_spam_data(i).name
    file_contents = readFile(file_name);
    [wordlist word_num]= extractWords(file_contents);
    words = [words wordlist];
    words_per_mail = [words_per_mail word_num];
end

spam_data = dir('data/spam/*');
for i=3:length(spam_data)
    file_name = spam_data(i).name
    file_contents = readFile(file_name);
    [wordlist word_num]= extractWords(file_contents);
    words = [words wordlist];
    words_per_mail = [words_per_mail word_num];
end


[w,~,idx] = unique(words);
numOccurrences = histcounts(idx,numel(w));
[rankOfOccurrences,rankIndex] = sort(numOccurrences,'descend');
wordsByFrequency = w(rankIndex);

vocabList = wordsByFrequency(1:2000);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% Part2 :with the vocabList, make data into X and y

% convert words into words_indexed(index determined by vocabList)
words_indexed = -ones(1,length(words));

for i=1:length(words)
    fprintf('%d',i)
    w = words(i);
    for j=1:length(vocabList)
        if strcmp(w,vocabList(j)) == 1
            words_indexed(i) = j;
            break;
        end
    end
end


%% 여기 전까지의 중간 결과 값들 words_data.mat에 있음.

m = length(non_spam_data) + length(spam_data) - 4;
Xall = zeros(m,2000);
yall = zeros(m,1);
yall(length(non_spam_data)-1:end) = 1;


% X의 각 row = 각 email, columns = 그 email에 vocabList의 단어들이 나왔는지 여부
count = 0;
for i=1:m
    word_num = words_per_mail(i);
    words_in_mail = words_indexed(count+1:count+word_num);
    for j=1:length(words_in_mail)
        if words_in_mail(j) ~= -1
            Xall(i,words_in_mail(j)) = 1;
        end
    end
    count = count + word_num
end
        

test_size = floor(m/5);
p = randperm(m);
Xtest = Xall(p(end-test_size+1:end),:);
ytest = yall(p(end-test_size+1:end));
X = Xall(p(1:end-test_size),:);
y = yall(p(1:end-test_size));

fprintf('\n X and y prepared. Press enter to continue.\n');
pause;

%% Part 3: Train SVM and test SVM

% train SVM
C = 0.1;
sigma = 0.1;
model = svmTrain(X, y, C, @linearKernel);
%model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));

p = svmPredict(model, X);

fprintf('Training Accuracy: %f\n', mean(double(p == y)) * 100);

% test
p = svmPredict(model, Xtest);

fprintf('Test Accuracy: %f\n', mean(double(p == ytest)) * 100);

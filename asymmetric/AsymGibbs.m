% Metropolis-within-Gibbs sampling for Asymmetric Gaussian mixture
clear;
fclose all;
% Change accuracy
digits(500);

% ------------ static parameters -------------
% log switch
dispLog = 1;
% Iteration number
T =10;
% multiplier for initial and new generated MU and SIGMA
multiplier = 0.001; % for real data
%multiplier = 0.01; % for synthetic data
% if using synthetic data
synthetic = 0;
% Observation amount
n = 400;
% initialization method: K-means or all zeros
kmeans = 1;
% Component number M
kmax=3;
% delete data file before each run
delFile = true;
% dimension
d = 3;
if synthetic % 2d and 3d only
    if d == 2
        muTmp = [-15,0; 15,0];
        sigmaTmp = [5,1; 1,1];% sigma lift and right for 2 dimensions
        sigmaTmp = [sigmaTmp; [1,5; 1,1]];
    elseif d == 3
        muTmp = [-30,0,0; 30,0,0;];
        sigmaTmp = [5,1; 1,1; 1,1];% sigma lift and right for 3 dimensions
        sigmaTmp = [sigmaTmp;[1,5; 1,1; 1,1]];
        %sigmaTmp = [sigmaTmp;[1,1; 1,1; 3,1]];
        %sigmaTmp = [sigmaTmp;[1,1; 1,1; 1,3]];
    end
    Utils.samplingFromAsymGaussian(n,d,delFile,muTmp,sigmaTmp);
    % generate dummy grouphat.csv
    grouphatTemp = [];
    while(size(grouphatTemp,1) < n)
        for i=1:size(muTmp,1)
            grouphatTemp = [grouphatTemp; ones(2^d,1)*(i-1)];
        end
    end
    delete 'grouphat.csv';
    csvwrite('grouphat.csv',grouphatTemp);
end
% initialize n observations
dataPoints = csvread('input.csv');
dataPoints = dataPoints + eps;

% model selection parameters
alpha = 3;
g = 0.3;
% RJMCMC parameters
rjmcmcEnabled = 1;
delta = 1;
lambda = 3;

% ---------- Feature Selection ----------------
featureSelectionFlag = 1;
if featureSelectionFlag
    % filter test data and remove obvious irrelavant features (values are the same)
    dataPoints = dataPoints(:,find(var(dataPoints)~=0));
end
d = size(dataPoints,2);

featureMu = mean(dataPoints);
% if synthetic
%     featureSigma = ones(1,d)*30;
% else
    %featureSigma = ones(1,d); % might be wrong
    featureSigma = var(dataPoints); % might be wrong
% end
% ---------------------------------------------------
% try different M
intLikeVector = [];
for kinit=2:2
    disp(['***************** k = ',num2str(kinit),' *****************']);
    disp('Initialization:');
    k = kinit;
    % re-initialize parameters
    muArray = [];
    sigmaArray = [];
    % feature selection
    featureRelevancy = [];
    if featureSelectionFlag
        featureRelevancy = ones(1,d)*0.5;
    end
    %----------------------------
    [membershipVector,muArray,sigmaArray] = Utils.kmeans_initialize(dataPoints,k);
    muArray = muArray + 0.1;
    if ~kmeans
        membershipVector = ones(1,k)./k;
        muArray = zeros(k,d);
        sigmaArray = ones(d*k,2)*0.5;
    end
    if synthetic
        mu = muTmp;
        sigma = sigmaTmp;
    else
        mu = muArray;
        sigma = sigmaArray;
    end
    
    % Gamma distribution initial parameters
    gamAlpha = 10000;
    gamBeta = 1;
    
    jumps = muArray;
    jumpCounter = 0;
    for i=1:T
        if dispLog || mod(i,10) == 0
            disp(['**** Round = ',num2str(i),' **** k =', num2str(k)]);
        end
        % re-initialize parameters
        newSigmaPriorProbTemp=[];
        oldSigmaPriorProbTemp=[];
        newSigmaProbTemp=[];
        oldSigmaProbTemp=[];
        %----------------------------
        % generate new parameters for background Gaussian distribution (feature selection)
        
        featureMuNew = mvnrnd(featureMu,eye(d)*multiplier) + eps;  % Unused
        featureSigmaNew = mvnrnd(featureSigma,eye(d)*multiplier) + eps;  % Unused
        % calculate feature Relevancy
        featureRelevancy = Utils.calculateFeatureRelevancy(d,k,dataPoints,muArray,sigmaArray,membershipVector,featureRelevancy,featureMu,featureSigma);
                
        % calculate membership vector
        gammaVector = ones(1,k) * delta;
        newMembershipVector = Utils.calculateMembershipVector(d,k,dataPoints,muArray,sigmaArray,membershipVector,gammaVector,featureRelevancy,featureMu,featureSigma);
        % generate new MUs from proposed 1D normal distribution (by dimensions)
        newMuArray = [];
        newSigmaArray = [];
        membershipMtx = Utils.calculateMembershipByPoint(d,k,dataPoints,muArray,sigmaArray,membershipVector,featureRelevancy,featureMu,featureSigma);
        for j=1:k
            % nj = sum(memMtxByComponent,1);
            dataMtxByComp = [];
            dataMtxByComp = dataPoints(membershipMtx(:,j) == 1,:);
            % calculate mean of data points
            %             if isempty(dataMtxByComp)
            newMuArray(j,:) = mvnrnd(muArray(j,:),eye(d)*multiplier) + eps; % if no datapoint is assigned to this component,use old MU to calculate new MU
            %             else
            %                 newMuArray(j,:) = mvnrnd(mean(dataMtxByComp,1),eye(d)*multiplier); % sigma for prior of MU is .1
            %             end
            % use Gamma distribution to sample sigma for every region of
            % asymmetric Gaussian
            %         for dimension = 1:d
            %             dataMtxByDim = dataMtxByComp(:,dimension);
            %             dataMtxLift = [];
            %             dataMtxRight = [];
            %             for dataRow = 1:size(dataMtxByDim,1)
            %                 % lift
            %                 if dataMtxByDim(dataRow) < muArray(j,dimension);
            %                     dataMtxLift = cat(1,dataMtxLift,dataMtxByDim(dataRow));
            %                 end
            %                 % right
            %                 if dataMtxByDim(dataRow) >= muArray(j,dimension);
            %                     dataMtxRight = cat(1,dataMtxRight,dataMtxByDim(dataRow));
            %                 end
            %             end
            %             % sample from Gamma distribution for both lift and right sides
            %             newSigmaLift = Utils.sampleFromGamma(gamAlpha,gamBeta,dataMtxLift,muArray(j,dimension));
            %             newSigmaright = Utils.sampleFromGamma(gamAlpha,gamBeta,dataMtxRight,muArray(j,dimension));
            %             newSigmaArray = cat(1,newSigmaArray,[newSigmaLift,newSigmaright]);
            %         end
        end
        % generate new SIGMAs from proposed 1D normal distribution (lift and right)
        for col=1:2
            newSigmaArray(:,col) = mvnrnd(sigmaArray(:,col),multiplier) + eps; % sigma for prior of SIGMA is defined by multiplier
            % compute absolute value for sigmas (if new sigma is less than 0, use sigma_old - sigma_new instead)
            for row=1:size(newSigmaArray,1);
                if newSigmaArray(row,col) < 0
                    newSigmaArray(row,col) = sigmaArray(row,col) - newSigmaArray(row,col);
                end
            end
        end
        
        %------------------------------------------------
        % compute acceptance ration
        R = 0;
        Rfeature = 0;
        
        % f(x | MUnew, SIGMAnew), here multiply all probability values to evaluate
        % the log likelihood
        newDataProb = sum(max(Utils.computeProbsAsymGaussian(d,k,dataPoints,newMuArray,newSigmaArray,newMembershipVector,featureRelevancy,featureMu,featureSigma),[],2));
        % multiply of all f(x | MUold, SIGMAold)
        oldDataProb = sum(max(Utils.computeProbsAsymGaussian(d,k,dataPoints,muArray,sigmaArray,membershipVector,featureRelevancy,featureMu,featureSigma),[],2));
        R = R + newDataProb - oldDataProb;  % likelihood ratio
        if dispLog
            disp(['Data R = ',num2str(newDataProb - oldDataProb)]);
        end
        
        % compute prior probs of Pi(MU) , assume MUs follow N([0,0],(multiplier*1000))
        newMuPriorProb = sum(log(mvnpdf(newMuArray,zeros(1,d),corr2cov(mean(dataPoints).^2 + multiplier*1000))));
        oldMuPriorProb = sum(log(mvnpdf(muArray,zeros(1,d),corr2cov(mean(dataPoints).^2 + multiplier*1000))));
        if abs(newMuPriorProb) ~= Inf && abs(oldMuPriorProb) ~= Inf
            R = R + newMuPriorProb - oldMuPriorProb;
            if dispLog
                disp(['Mu Prior R = ',num2str(newMuPriorProb - oldMuPriorProb)]);
            end
        end
        % compute prior probs of Pi(SIGMA), assume both lift and right parts of
        % each dimension follow N(0,multiplier*1000)
        for j=1:2
            newSigmaPriorProbTemp(:,j) = log(mvnpdf(newSigmaArray(:,j),0,corr2cov(repmat(featureSigma,1,k).^2 + multiplier*1000)));
            oldSigmaPriorProbTemp(:,j) = log(mvnpdf(sigmaArray(:,j),0,corr2cov(repmat(featureSigma,1,k).^2 + multiplier*1000)));
        end
        newSigmaPriorProb = sum(sum(newSigmaPriorProbTemp));
        oldSigmaPriorProb = sum(sum(oldSigmaPriorProbTemp));
        if abs(newSigmaPriorProb) ~= Inf && abs(oldSigmaPriorProb) ~= Inf
            R = R + newSigmaPriorProb - oldSigmaPriorProb;
            if dispLog
                disp(['Sigma Prior R = ',num2str(newSigmaPriorProb - oldSigmaPriorProb)]);
            end
        end

        % compute proposal probs of MUs (old - new)
        for j=1:k
            oldMuProbTemp(j,:) = log(mvnpdf(muArray(j,:),newMuArray(j,:),eye(d)));
            newMuProbTemp(j,:) = log(mvnpdf(newMuArray(j,:),muArray(j,:),eye(d)));
        end
        oldMuProb = sum(oldMuProbTemp);
        newMuProb = sum(newMuProbTemp);
        if abs(oldMuProb) ~= Inf && abs(newMuProb) ~= Inf
            R = R + oldMuProb - newMuProb;
            if dispLog
                disp(['Mu R = ',num2str(oldMuProb - newMuProb)]);
            end
        end
        % compute proposal probs of SIGMAs (old - new)
        for j=1:2
            oldSigmaProbTemp(:,j) = log(mvnpdf(sigmaArray(:,j),newSigmaArray(:,j),1));
            newSigmaProbTemp(:,j) = log(mvnpdf(newSigmaArray(:,j),sigmaArray(:,j),1));
        end
        oldSigmaProb = sum(sum(oldSigmaProbTemp));
        newSigmaProb = sum(sum(newSigmaProbTemp));
        if abs(oldSigmaProb) ~= Inf && abs(newSigmaProb) ~= Inf
            R = R + oldSigmaProb - newSigmaProb;
            if dispLog
                disp(['Final R = ',num2str(R)]);
            end
        end
        % generate random number from [0,1]
        u = log(rand(1));
        paramAcpt = 0;

        if R > u
            paramAcpt = 1;
            muArray = newMuArray;
            sigmaArray = newSigmaArray;
            membershipVector = newMembershipVector;
            jumps=cat(1,jumps,muArray);
            jumpCounter = jumpCounter + 1;
            if dispLog
                disp(['Jump accepted!  k = ',num2str(k)]);
            end
        end
        if rjmcmcEnabled %&& paramAcpt
            % Apply RJMCMC steps only if classic Gibbs steps have been
            % accepted
            
            % define temp variables
            
            % --------------- Split step -------------------------
%             Bk = Utils.calculateBk(k,kmax);
%             if Bk > rand(1)
%                 indexSelected = ceil(rand(1)*k);
%                 
%                 %generate 3-dimensional random vector
%                 randomVector = [];
%                 randomVector(1) = betarnd(2,2);
%                 randomVector(2) = betarnd(2,2);
%                 randomVector(3) = betarnd(1,1);
%                 
%                 %remove weight,mu and sigma of selected J
%                 tempWeights = [];
%                 tempMus = [];
%                 tempSigmas = [];
%                 for j=1:k
%                     if j == indexSelected
%                         continue;
%                     end
%                     tempWeights = [tempWeights newMembershipVector(j)];
%                     tempMus = cat(1,tempMus,newMuArray(j,:));
%                     tempSigmas = cat(1,tempSigmas,newSigmaArray((j-1)*d+1:(j-1)*d+d,:));
%                 end
%                 %add weights of J1 and J2
%                 tempWeights = [tempWeights newMembershipVector(indexSelected)*randomVector(1) newMembershipVector(indexSelected)*(1 - randomVector(1))];
%                 %calculate MUs for J1 and J2
%                 muOld = newMuArray(indexSelected,:);
%                 sigmasOld = newSigmaArray((indexSelected-1)*d+1:(indexSelected-1)*d+d,:);
%                 muJ1 = muOld - randomVector(2)* transpose(mean(sigmasOld,2)) * sqrt(tempWeights(k+1)./tempWeights(k));
%                 muJ2 = muOld + randomVector(2)* transpose(mean(sigmasOld,2)) * sqrt(tempWeights(k)./tempWeights(k+1));
%                 tempMus = [tempMus;muJ1;muJ2];
%                 %check if muJ1 and muJ2 satisfy eq 9 (no other component between J1 and J2)
%                 distanceArrayJ1 = sum(bsxfun(@minus,tempMus,tempMus(k,:)).^2,2);
%                 distanceArrayJ2 = sum(bsxfun(@minus,tempMus,tempMus(k+1,:)).^2,2);
%                 indexClosestJ1 = find(distanceArrayJ1 == min(distanceArrayJ1(distanceArrayJ1 > 0 ,:)));
%                 indexClosestJ2 = find(distanceArrayJ2 == min(distanceArrayJ2(distanceArrayJ2 > 0 ,:)));
%                 if (indexClosestJ1 == k+1) || (indexClosestJ2 == k)
%                     %calculate sigmas for J1 and J2
%                     sigmasJ1 = sigmasOld * sqrt(randomVector(3)*(1-randomVector(2).^2)*newMembershipVector(indexSelected)./tempWeights(k));
%                     sigmasJ2 = sigmasOld * sqrt((1-randomVector(3))*(1-randomVector(2).^2)*newMembershipVector(indexSelected)./tempWeights(k+1));
%                     tempSigmas = [tempSigmas ; sigmasJ1 ; sigmasJ2];
%                     %calculate likelihood ratio P(x|Theta_new)/P(x|Theta_old)
%                     likeliRatio = sum(max(Utils.computeProbsAsymGaussian(d,k+1,dataPoints,tempMus,tempSigmas,tempWeights,featureRelevancy,featureMu,featureSigma),[],2))./sum(max(Utils.computeProbsAsymGaussian(d,k,dataPoints,newMuArray,newSigmaArray,newMembershipVector,featureRelevancy,featureMu,featureSigma),[],2));
%                     
%                     %split acceptance ratio
%                     if Utils.calculateMergeSpliteProbability('split',randomVector,d,k,kmax,likeliRatio,alpha,g,lambda,delta,k,k+1,dataPoints,tempMus,tempSigmas,tempWeights,muOld,sigmasOld,featureRelevancy,featureMu,featureSigma) > log(rand(1)) % log(A) > log(rand(1))
%                         disp('split accepted!');
%                         newMembershipVector = tempWeights ;
%                         newMuArray = tempMus;
%                         newSigmaArray = tempSigmas;
%                         %increase k by 1
%                         k = k+1;
%                         %update parameters
%                         muArray = newMuArray;
%                         sigmaArray = newSigmaArray;
%                         membershipVector = newMembershipVector;
%                     end
%                 end
%             end
%             %--------------- Split step ends -------------------------
%             %--------------- Merge step -------------------------
%             Bk = Utils.calculateBk(k,kmax);
%             Dk = 1 - Bk;
%             if Dk > rand(1)
%                 indexSelected = ceil(rand(1)*k);
%                 %calculate the distances between Mus and select the
%                 %closest one to the selected component Mu
%                 distanceArray = sum(bsxfun(@minus,newMuArray,newMuArray(indexSelected,:)).^2,2);
%                 indexClosest = find(distanceArray == min(distanceArray(distanceArray > 0 ,:)));
%                 %calculate new parameters for merged component j*
%                 muJ1 = newMuArray(indexSelected,:);
%                 muJ2 = newMuArray(indexClosest,:);
%                 membershipJ1 = newMembershipVector(indexSelected);
%                 membershipJ2 = newMembershipVector(indexClosest);
%                 membershipJnew = membershipJ1 + membershipJ2;
%                 muJnew = ( membershipJ1*muJ1 + membershipJ2 * muJ2 )./ membershipJnew;
%                 %calculate sigma dimension by dimension
%                 sigmaArrayJnew = [];
%                 sigmaArrayJ1 = newSigmaArray((indexSelected-1)*d+1:(indexSelected-1)*d+d,:);
%                 sigmaArrayJ2 = newSigmaArray((indexClosest-1)*d+1:(indexClosest-1)*d+d,:);
%                 for j=1:d
%                     for col=1:2
%                         sigmaArrayJnew(j,col) = sqrt((membershipJ1*(muJ1(j).^2 + sigmaArrayJ1(j,col).^2) + membershipJ2*(muJ2(j).^2 + sigmaArrayJ2(j,col).^2))./membershipJnew - muJnew(j).^2);
%                     end
%                 end
%                 %calculate acceptance ration of merge step
%                 %remove weights,mus and sigmas of J1 and J2
%                 tempWeights = [];
%                 tempMus = [];
%                 tempSigmas = [];
%                 for j=1:k
%                     if j == indexSelected || j == indexClosest
%                         continue;
%                     end
%                     tempWeights = [tempWeights newMembershipVector(j)];
%                     tempMus = cat(1,tempMus,newMuArray(j,:));
%                     tempSigmas = cat(1,tempSigmas,newSigmaArray((j-1)*d+1:(j-1)*d+d,:));
%                 end
%                 %add new weight, mu and sigma for Jnew
%                 tempWeights = [tempWeights membershipJnew];
%                 tempMus = cat(1,tempMus,muJnew);
%                 tempSigmas = cat(1,tempSigmas, sigmaArrayJnew);
%                 %calculate likelihood ratio P(x|Theta_old)/P(x|Theta_new)
%                 likeliRatio = sum(max(Utils.computeProbsAsymGaussian(d,k,dataPoints,newMuArray,newSigmaArray,newMembershipVector,featureRelevancy,featureMu,featureSigma),[],2)) ./ sum(max(Utils.computeProbsAsymGaussian(d,k-1,dataPoints,tempMus,tempSigmas,tempWeights,featureRelevancy,featureMu,featureSigma),[],2));
%                 %generate 3-dimensional random vector
%                 randomVector = [];
%                 randomVector(1) = betarnd(2,2);
%                 randomVector(2) = betarnd(2,2);
%                 randomVector(3) = betarnd(1,1);
%                 if -Utils.calculateMergeSpliteProbability('merge',randomVector,d,k-1,kmax,likeliRatio,alpha,g,lambda,delta,indexSelected,indexClosest,dataPoints,newMuArray,newSigmaArray,newMembershipVector,muJnew,sigmaArrayJnew,featureRelevancy,featureMu,featureSigma) > log(rand(1)) % log(A^-1) > log(rand(1)) -> -log(A) > log(rand(1))
%                     disp('merge accepted!');
%                     newMembershipVector = tempWeights ;
%                     newMuArray = tempMus;
%                     newSigmaArray = tempSigmas;
%                     %reduce k by 1
%                     k = k-1;
%                     %update parameters
%                     muArray = newMuArray;
%                     sigmaArray = newSigmaArray;
%                     membershipVector = newMembershipVector;
%                 end
%             end
%             % --------------- Merge step ends -------------------------
            % --------------- Birth step ---------------------------
            birthFlag = 0;
            Bk = Utils.calculateBk(k,kmax);
            if Bk > rand(1)
                % generate parameters for new component Jnew
                weightJnew = betarnd(1,k);
                % generate MUnew from N(xi,kappa^-1)
                MUJnew = mvnrnd(mean(dataPoints),diag((max(dataPoints) - min(dataPoints)).^2));
                % SIGMAj^-2 ~ Ga(alpha,beta) and beta also follow Ga(g,h). g is a
                % constant and h is a small multiple of 1./R^2
                % first, sample beta from Ga(g,h)
                sigmaJnew = [];
                for j=1:d
                    for col=1:2 % lift and right part of each dimension
                        if col ==1
                            dataMtxJnewPartial = dataPoints(dataPoints(:,j) <= MUJnew(j),:);
                        else
                            dataMtxJnewPartial = dataPoints(dataPoints(:,j) > MUJnew(j),:);
                        end
                        if size(dataMtxJnewPartial,1) == 0
                            sigmaJnew(j,col) = 1;
                        else
                            RJnew = 2 * max(abs(MUJnew(j) - dataMtxJnewPartial(:,j)));
                            BETAJnew = gamrnd(g,200*g/(alpha*RJnew.^2));
                            BETAJnew = max(0.1,BETAJnew);
                            sigmaJnew(j,col) = sqrt(1./gamrnd(alpha,BETAJnew));
                        end
                    end
                end
                % add new weight, mu and sigma for Jnew
                tempWeights = [];
                tempMus = [];
                tempSigmas = [];
                tempWeights = [newMembershipVector*(1 - weightJnew) weightJnew];
                tempMus = cat(1,newMuArray,MUJnew);
                tempSigmas = cat(1,newSigmaArray, sigmaJnew);
                if Utils.calculateBirthDeathProbability('birth',d,k,kmax,dataPoints,lambda,delta,weightJnew,newMuArray,newSigmaArray,newMembershipVector,featureRelevancy,featureMu,featureSigma) > log(rand(1))
                    disp('birth accepted!');
                    newMembershipVector = tempWeights ;
                    newMuArray = tempMus;
                    newSigmaArray = tempSigmas;
                    % increase k by 1
                    k = k+1;
                    % update parameters
                    muArray = newMuArray;
                    sigmaArray = newSigmaArray;
                    membershipVector = newMembershipVector;
                    birthFlag =1;
                end
            end
            % --------------- Birth step ends ---------------------------
            % --------------- Death step ---------------------------
             Bk = Utils.calculateBk(k,kmax);
             Dk = 1 - Bk;
             %if birthFlag || Dk > rand(1)
                % randomly select an empty component
                membershipMtx = Utils.calculateMembershipByPoint(d,k,dataPoints,newMuArray,newSigmaArray,newMembershipVector,featureRelevancy,featureMu,featureSigma);
                if ~isempty(find(sum(membershipMtx,1)==0))
                    emptyIndexes = find(sum(membershipMtx,1)==0);
                    indexSelected = ceil(rand(1)*size(emptyIndexes,2));
                    indexSelected = emptyIndexes(indexSelected);
                    % remove selected component
                    % remove weight,mu and sigma of selected J
                    tempWeights = [];
                    tempMus = [];
                    tempSigmas = [];
                    for j=1:k
                        if j == indexSelected
                            continue;
                        end
                        tempWeights = [tempWeights newMembershipVector(j)];
                        tempMus = cat(1,tempMus,newMuArray(j,:));
                        tempSigmas = cat(1,tempSigmas,newSigmaArray((j-1)*d+1:(j-1)*d+d,:));
                    end
                    % normalize weights so that sum of weights equals to 1
                    tempWeights = tempWeights ./(1-newMembershipVector(indexSelected));
                    
                    %if -Utils.calculateBirthDeathProbability('death',d,k-1,kmax,dataPoints,lambda,delta,newMembershipVector(indexSelected),tempMus,tempSigmas,tempWeights,featureRelevancy,featureMu,featureSigma) > log(rand(1))
                        disp('death accepted!');
                        newMembershipVector = tempWeights ;
                        newMuArray = tempMus;
                        newSigmaArray = tempSigmas;
                        % reduce k by 1
                        k = k-1;
                        % update parameters
                        muArray = newMuArray;
                        sigmaArray = newSigmaArray;
                        membershipVector = newMembershipVector;
                    %end
                end
             %end
            % --------------- Death step ends ---------------------------
        end
        
    end
    if dispLog
        % display results
        disp('results:');
        disp('Mu Array :');
        disp(muArray);
        disp('Sigma Array :');
        disp(sigmaArray);
    end
    
    if synthetic && k == 2 && size(mu,1) == 2
        % calculate Euclidean distance between calculated mus and original
        % mus
        disp('Euclidean distance of calculated and original MUs and SIGMAs :')
        for i=1:size(mu,1)
            disp(['For No.:',num2str(i),' cluster:']);
            %             disp('original Mu:');
            %             disp(mu(i,:));
            %             disp('calculated Mu:');
            %             disp(muArray(i,:));
            disp(['Euclidean distance of Mu : ',num2str(sqrt(sum((muArray(i,:) - mu(i,:)).^2)))]);
            disp(['Euclidean distance of Mu : ',num2str(sqrt(sum((muArray(size(mu,1)-i+1,:) - mu(size(mu,1)-i+1,:)).^2)))]);
            %             disp('original Sigma:');
            %             disp(sigma((i-1)*d+1:(i-1)*d+d,:));
            %             disp('calculated Sigma:');
            %             disp(sigmaArray((i-1)*d+1:(i-1)*d+d,:));
            disp(['Euclidean distance of Sigma : ',num2str(sqrt(sum((sigma((i-1)*d+1:(i-1)*d+d,:)-sigmaArray((i-1)*d+1:(i-1)*d+d,:)).^2)))]);
            disp(['Euclidean distance of Sigma : ',num2str(sqrt(sum((sigma((size(mu,1)-i)*d+1:(size(mu,1)-i)*d+d,:)-sigmaArray((size(mu,1)-i)*d+1:(size(mu,1)-i)*d+d,:)).^2)))]);
        end
    end
    
    % Model selection: use marginal likelihood to estimate component number
    % and ' out the maxmium value of P(X|M)
    integratedLikelihood = 0;
    np = (2*d+1)*k; % number of parameters need to be evaluated
    integratedLikelihood = integratedLikelihood + (np./2)*log(2*pi);
    
    % Log-likelihood : P(X|theta,M)
    integratedLikelihood = integratedLikelihood + sum(max(Utils.computeProbsAsymGaussian(d,k,dataPoints,muArray,sigmaArray,membershipVector,featureRelevancy,featureMu,featureSigma),[],2));
    membershipMtx = Utils.calculateMembershipByPoint(d,k,dataPoints,muArray,sigmaArray,membershipVector,featureRelevancy,featureMu,featureSigma);
    Pj = membershipMtx * transpose(membershipVector);
    integratedLikelihood = integratedLikelihood + sum(log(Pj));
    
    % hessian matrix (determinant of covariance matrix)
    sigLift = [];
    sigRight = [];
    for i=1:k
        for j=1:d
            sigLift(i,j)=sigmaArray((i-1)*d+j,1);
            sigRight(i,j)=sigmaArray((i-1)*d+j,2);
        end
    end
    parameterMtx = muArray;
    parameterMtx = cat(2,parameterMtx,mean(sigLift,2));
    parameterMtx = cat(2,parameterMtx,mean(sigRight,2));
    hessianMtx = det(cov(parameterMtx));
    if hessianMtx > 0 % determinant of hessian matrix could be 0 or negative, ignore this case
        integratedLikelihood = integratedLikelihood + (1./2)*log(det(cov(parameterMtx)));
    end
    
    % Prior of parameter based on component number M: Log(P(Theta|M))
    dataMtxByComp={};
    for i=1:k
        dataMtxByComp{i} = dataPoints(membershipMtx(:,i) == 1,:);
        if ~isempty(dataMtxByComp{i})
            % MUj ~ N(mean(datapoints),interval^2)
            intvVectorByDimension = max(dataMtxByComp{i},[],1) - min(dataMtxByComp{i},[],1);
            intvVectorByDimension = intvVectorByDimension + 0.1;
            integratedLikelihood = integratedLikelihood + log(mvnpdf(muArray(i,:),mean(dataMtxByComp{i},1),diag(intvVectorByDimension.^2)));
            
            
            % SIGMAj^-2 ~ Ga(alpha,beta) and beta also follow Ga(g,h). g is a
            % constant and h is a small multiple of 1./R^2
            % first, sample beta from Ga(g,h)
            % use datapoints belong to either lift or right side to get
            % more proper accessment of asymmetric SIGMAs
            tempIntegratedLikelihood = 0.0;
            for j=1:d
                dataMtxLift = dataMtxByComp{i}(dataMtxByComp{i}(:,j) <= muArray(i,j),:);
                RLift = 2 * max(muArray(i,j) - dataMtxLift(:,j));
                betaLift = gamrnd(g,200*g/(alpha*RLift.^2));
                betaLift = max(0.1,betaLift);
                if gampdf(sigmaArray((i-1)*d+j,1).^(-2),alpha,betaLift) > 0
                    tempIntegratedLikelihood = tempIntegratedLikelihood + log(gampdf(sigmaArray((i-1)*d+j,1).^(-2),alpha,betaLift));
                end
                dataMtxRight = dataMtxByComp{i}(dataMtxByComp{i}(:,j) > muArray(i,j),:);
                RRight = 2 * max(dataMtxRight(:,j) - muArray(i,j));
                betaRight = gamrnd(g,200*g/(alpha*RRight.^2));
                betaRight = max(0.1,betaRight);
                if gampdf(sigmaArray((i-1)*d+j,2).^(-2),alpha,betaRight) > 0
                    tempIntegratedLikelihood = tempIntegratedLikelihood + log(gampdf(sigmaArray((i-1)*d+j,2).^(-2),alpha,betaRight));
                end
            end
            integratedLikelihood = integratedLikelihood + tempIntegratedLikelihood;
        end
    end
    
    % calculate confusion matrix
    group = [];
    for i=1:size(membershipMtx,2)
        group = [group membershipMtx(:,i)*i];
    end
    group = sum(group,2);
    Utils.confusionMatrix(group);
    
    if d <=3
        Utils.plotDensityRegion(d,dataPoints, muArray,sigmaArray,kinit,group);
        % plot jump traces (don't apply to RJMCMC)
        if ~rjmcmcEnabled
            Utils.plotJumpLine(jumps,d,k);
        end
        Utils.drawAnnotations(size(dataPoints,1),T,jumpCounter,integratedLikelihood);
        hold off;
    end
    disp('marginalLikelihood = ');
    disp(integratedLikelihood);
    
    if featureSelectionFlag
        disp('Feature Relevancy = ');
        disp(featureRelevancy);
    end
end
if d <=3
    grouphat = csvread('grouphat.csv');
    Utils.plotDensityRegion(d,dataPoints, mu,sigma,0,grouphat+1);
end
classdef Utils
    %UTILS Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods(Static)
        function confusionMatrix(membershipMtx)
            % calculate confusion matrix
            grouphat = csvread('grouphat.csv');
            confMtx = [];
            for i=1:size(membershipMtx,2)
                confMtx = [confMtx membershipMtx(:,i)*i];
            end
            disp('Confusion Matrix : ');
            confMtx = confusionmat(sum(confMtx,2),grouphat+1)
            disp('Accuracy : ');
            accur = trace(confMtx)/ sum(sum(confMtx))
            1 - accur
        end
        
        %         function GMM_initialize(data,M)
        %             GMModel = fitgmdist(data,M,'RegularizationValue',eps);
        %             idx = cluster(GMModel,data);
        %             if M ==2
        %                 disp('Confusion Matrix for GMM : ')
        %                 Utils.confusionMatrix(idx);
        %             end
        %         end
        
        
        function [a,mu,sigmaMtx]=kmeans_initialize(data,M)
            [nbVar, nbData] = size(data);
            % K-means
            [Data_id,centers]=kmeans(data,M);
            %if M ==2
                disp('Confusion Matrix for K-means : ')
                Utils.confusionMatrix(Data_id);
            %end
            mu=centers;
            sigmaL=[];
            sigmaR=[];
            for i=1:M
                idtmp = find(Data_id==i);
                a(i) = length(idtmp);
                
                dataTmp = data(idtmp,:);
                % calculate asymmetric standard deviation, data - mean(data)
                for j=1:size(dataTmp,1)
                    dataTmp(j,:) = dataTmp(j,:) - mu(i,:);
                end
                % for every dimension, calculate sigmaL and sigmaR
                for j=1:size(dataTmp,2)
                    % sigmaL should be calculated based on those
                    % observations that Xi - MUi < 0
                    % standard deviation SD = sqrt(sum((x-MU).^2)/N)
                    dataTmpLeft = dataTmp(dataTmp(:,j) < 0,j);
                    if size(dataTmpLeft,1) > 0
                        sigmaL = [sigmaL sqrt(sum(dataTmpLeft.^2)/size(dataTmpLeft,1))];
                    else
                        sigmaL = [sigmaL 0];
                    end
                    % in contrast, sigmaR for Xi - MUi >= 0
                    dataTmpRight = dataTmp(dataTmp(:,j) >= 0,j);
                    if size(dataTmpRight,1) > 0
                        sigmaR = [sigmaR sqrt(sum(dataTmpRight.^2)/size(dataTmpRight,1))];
                    else
                        sigmaR = [sigmaR 0];
                    end
                end
            end
            sigmaL = sigmaL + eps(1);
            sigmaR = sigmaR + eps(1);
            sigmaMtx = [transpose(sigmaL) transpose(sigmaR)];
            a = a ./ sum(a);
            
            % GMM
            data=data';
            Sigma= [];
            for i=1:M
                idtmp = find(Data_id==i);
                Sigma(:,:,i)=cov([data(:,idtmp) data(:,idtmp)]');
                %Add a tiny variance to avoid numerical instability
                Sigma(:,:,i) = Sigma(:,:,i) + 1E-5.*diag(ones(nbData,1));
            end   
            obj = gmdistribution(mu,Sigma,a);
            idx = cluster(obj,data');
            %if M ==2
                disp('Confusion Matrix for GMM : ')
                Utils.confusionMatrix(idx);
            %end
        end
        
        function x = samplingFromAsymGaussian(n,d,delFile,mus,sigma)
            if delFile
                delete 'input.csv';
            end
            if exist('input.csv','file')
                disp('File exists!');
                x=csvread('input.csv');
            else
                x =[];
                while size(x,1) < n
                    for i=1:size(mus,1)
                        % sampling from  2*d*d 1D independent distributions
                        % recursively
                        x = cat (1,x,Utils.samplingByDimensions(mus(i,:),sigma((i-1)*d+1:(i-1)*d+d,:),[],[],1));
                    end
                end
                csvwrite('input.csv',x);
            end
        end
        
        function x = samplingByDimensions(mu,sigma,selectedSigmaVector,positionVector,dimensionIndex)
            x=[];
            if dimensionIndex > size(mu,2)
                % sample here
                flag = true;
                while flag
                    obj = gmdistribution(mu,corr2cov(selectedSigmaVector));
                    x = random(obj);
                    flag = false;
                    for i = 1 : size(positionVector,2)
                        % check validity
                        if (x(i) >= mu(i) && ~positionVector(i)) || (x(i) < mu(i) && positionVector(i))
                            flag = true;
                            break;
                        end
                    end
                end
                return;
            end
            % lift part
            tempSelectedSigmaVector = [selectedSigmaVector,sigma(dimensionIndex,1)];
            tempPositionVector = [positionVector, 0];
            x = cat(1,x,Utils.samplingByDimensions(mu,sigma,tempSelectedSigmaVector,tempPositionVector,dimensionIndex + 1));
            % right part
            tempSelectedSigmaVector = [selectedSigmaVector,sigma(dimensionIndex,2)];
            tempPositionVector = [positionVector, 1];
            x = cat(1,x,Utils.samplingByDimensions(mu,sigma,tempSelectedSigmaVector,tempPositionVector,dimensionIndex + 1));
        end
        
        function x = calculateMembershipVector(d,k,dataPoints,muArray,sigmaArray,membershipVector,gammaVector,featureRelevancy,featureMu,featureSigma)
            
            x = Utils.calculateMembershipByPoint(d,k,dataPoints,muArray,sigmaArray,membershipVector,featureRelevancy,featureMu,featureSigma);
            % gammaVector = ones(1,k)*50;
            x = sum(x);
            x = Utils.drchrnd(x + gammaVector,1);
        end
        
        function x = calculateMembershipByPoint(d,k,dataPoints,muArray,sigmaArray,membershipVector,featureRelevancy,featureMu,featureSigma)
            x = Utils.computeProbsAsymGaussian(d,k,dataPoints,muArray,sigmaArray,membershipVector,featureRelevancy,featureMu,featureSigma);
            
            % find largest probability for all components and set it to 1
            [M I] = max(x,[],2);
            
            % use maximum indexes I to generate membership vector
            % (dirichlet distribution)
            x= zeros(size(x));
            for i=1:size(x,1)
                x(i,I(i)) = 1;
            end
        end
        
        % using gamma to simulate dirichlet distribution
        function r = drchrnd(a,n)
            p = length(a);
            r = gamrnd(repmat(a,n,1),1,n,p);
            r = r ./ repmat(sum(r,2),1,p);
        end
        
        % sample from Gamma distribution by dimension
        function x = sampleFromGamma(alpha,betaVal,dataMtx,mu)
            ni = size(dataMtx,1);
            alphaNew = alpha + ni;
            betaNew = betaVal;
            if ni > 0
                betaNew = betaNew + transpose(dataMtx - mu)*(dataMtx - mu);
            end
            disp('betaNew = ');
            disp(betaNew);
            x = gamrnd(alphaNew,1./betaNew);
        end
        
        % calculate probabilities from asymmetricGaussian
        function x = computeProbsAsymGaussian(d,k,dataPoints,muArray,sigmaArray,membershipVector,featureRelevancy,featureMu,featureSigma)
            x=[];
            for i=1:size(dataPoints)
                point = dataPoints(i,:);
                for j=1:k
                    mu = muArray(j,:);
                    sigmaVector = [];
                    for l=1:d
                        if point(l) < mu(l)  % calculate lookup vector of sigma for component k
                            sigmaVector = [sigmaVector,sigmaArray((j-1)*d+l,1)];
                        else
                            sigmaVector = [sigmaVector,sigmaArray((j-1)*d+l,2)];
                        end
                    end
                    %                     disp('mu:');
                    %                     disp(mu);
                    %                     disp('lookup vector:');
                    %                     disp(sigmaVector);
                    % calculate probability for component k
                    x(i,j) = Utils.asymmetricGaussianPDF(d,k,point,mu,sigmaVector,sigmaArray((j-1)*d+1:j*d,:),membershipVector(j),featureRelevancy,featureMu,featureSigma);
                end
            end
        end
        
        % calculate PDF for a specific datapoint
        function x = asymmetricGaussianPDF(d,k,dataPoint,mu,sigmaVector,sigmaMatrix,weight,featureRelevancy,featureMu,featureSigma)
            x = log(weight);
            sigmaSum = sum(sigmaMatrix,2);
            for i=1:d
                %                 disp(-log(sigmaSum(i,:)));
                %                 disp((dataPoint(i)-mu(i)).^2);
                %                 disp(2*sigmaVector(i).^2);
                %x = x./(sigmaSum(i,:)*exp((dataPoint(i)-mu(i)).^2./2*sigmaVector(i).^2));
                % ln(1/sum of sigma) + ln(exp part)
                xTemp = -log(sigmaSum(i,:)) - (dataPoint(i)-mu(i)).^2./(2*sigmaVector(i).^2);
                if isempty(featureRelevancy)
                    x = x + xTemp;
                else
                    % log(w*P + (1-w)*P') = log(P*(w + (1-w)*P'/P)) =
                    % log(P) + log(w + (1-w)*P'/P) = 
                    % log(P) + log(w + (1-w)*exp(log(P') - log(P)))
                    %x = x + xTemp + log(featureRelevancy(i) + (1-featureRelevancy(i)*exp(log(mvnpdf(dataPoint(i),featureMu(i),featureSigma(i))) - xTemp)));              
                    if featureRelevancy(i)
                        x = x + xTemp;
                    else
                        x = x + log(mvnpdf(dataPoint(i),featureMu(i),featureSigma(i)));
                    end
                end
            end
        end
        
        % feature selection, calculate feature relevancy
        function x = calculateFeatureRelevancy(d,k,dataPoints,muArray,sigmaArray,membershipVector,featureRelevancy,featureMu,featureSigma)              
            x = zeros(1,d);
            membershipMtx = Utils.computeProbsAsymGaussian(d,k,dataPoints,muArray,sigmaArray,membershipVector,[],[],[]);
            
            % find largest probability for all components and set it to 1
            [M I] = max(membershipMtx,[],2);
            
            % use maximum indexes I to generate membership vector
            % (dirichlet distribution)
            membershipMtx= zeros(size(membershipMtx));
            for i=1:size(membershipMtx,1)
                membershipMtx(i,I(i)) = 1;
            end
            
            likelihoodTemp=zeros(1,d);
            for j=1:k
                dataMtxByComp = [];
                dataMtxByComp = dataPoints(membershipMtx(:,j) == 1,:);
                
                mu = muArray(j,:);              
                for n = 1:size(dataMtxByComp,1)
                    point = dataMtxByComp(n,:);
                    for l=1:d
                        if point(l) < mu(l)  % calculate lookup vector of sigma for component k
                            selectedSigma = sigmaArray((j-1)*d+l,1);
                        else
                            selectedSigma = sigmaArray((j-1)*d+l,2);
                        end
                        sigmaSum = sigmaArray((j-1)*d+l,1) + sigmaArray((j-1)*d+l,2);
                        likelihoodTemp(l) = likelihoodTemp(l) + log(membershipVector(j)) -log(sigmaSum) - (point(l)-mu(l)).^2./(2*selectedSigma.^2);
                    end
                end
            end
            
            for i=1:d
                % w = P / (P + P') = 1 / (1 + P'/P) = (1+exp(log(P')-log(P)))^-1
                likelihoodBackground = sum(log(mvnpdf(dataPoints(:,i),featureMu(i),featureSigma(i)))); % P'
                relevancyTemp = likelihoodBackground - likelihoodTemp(i);
                relevancyTemp = exp(relevancyTemp);
                % workaround: normalize relevancy to 0 or 1
                xTemp = (1 + relevancyTemp)^-1;
                if xTemp >= .5
                    xTemp = 1;
                else
                    xTemp = 0;
                end
                x(i) = xTemp;
            end      
        end
        
        function x = generateFromNormaldistribution(mu,sigma)
            x = normrnd(mu,sigma);
        end
        
        
        function plotDensityRegion(d,dataPoints,mus,sigmas,kinit,group)
            
            if kinit == 0
                figure('Name','Original Mixture','NumberTitle','off');
            else
                figure('Name',strcat('Initial K is  ',num2str(kinit)),'NumberTitle','off');
            end
            colors = 'rgbymcwk';
            if size(mus,2) == 2 %plot here (2d)
                gscatter(dataPoints(:,1),dataPoints(:,2),group);
                xlabel('X');
                ylabel('Y');
            elseif size(mus,2) == 3 %plot here (3d)
                for i=1:size(mus,1)
                    scatter3(dataPoints(group == i ,1), dataPoints(group == i,2),dataPoints(group == i,3),'filled','markerFaceColor',colors(i));
                    hold on;
                end
                xlabel('X');
                ylabel('Y');
                zlabel('Z');
            end
            hold on;
            for i=1:size(mus,1)
                Utils.plotAsymmetricDensity(mus(i,:),sigmas((i-1)*d + 1: i*d,:),[],[],1);
            end
        end
        
        function plotAsymmetricDensity(mu,sigma,selectedSigmaVector,positionVector,dimensionIndex)
            if dimensionIndex > size(mu,2)
                if size(mu,2) == 2 %plot here (2d)
                    muX = mu(1);
                    muY = mu(2);
                    if positionVector(1)
                        x = muX:.1:(muX+10); %// x axis
                    else
                        x = muX-10:.1:muX; %// x axis
                    end
                    
                    if positionVector(2)
                        y = muY:.1:(muY+10); %// x axis
                    else
                        y = muY-10:.1:muY; %// x axis
                    end
                    
                    [X Y] = meshgrid(x,y); %// all combinations of x, y
                    dataMtx = [X(:) Y(:)];
                    
                    Z = mvnpdf(dataMtx,mu,corr2cov(selectedSigmaVector)); %// compute Gaussian pdf
                    Z = reshape(Z,size(X)); %// put into same size as X, Y
                    contour(X,Y,Z) %, axis([-10 10 -10 10]);  %// contour plot; set same scale for x and y.
                    %surf(X,Y,Z) %// . or 3D plot
                elseif size(mu,2) == 3 %plot here (3d)
                    plot_gaussian_ellipsoid(mu, corr2cov(selectedSigmaVector), 2, [], gca, positionVector);
                end
                hold on;
                return;
            end
            % lift part
            tempSelectedSigmaVector = [selectedSigmaVector,sigma(dimensionIndex,1)];
            tempPositionVector = [positionVector, 0];
            Utils.plotAsymmetricDensity(mu,sigma,tempSelectedSigmaVector,tempPositionVector,dimensionIndex + 1);
            % right part
            tempSelectedSigmaVector = [selectedSigmaVector,sigma(dimensionIndex,2)];
            tempPositionVector = [positionVector, 1];
            Utils.plotAsymmetricDensity(mu,sigma,tempSelectedSigmaVector,tempPositionVector,dimensionIndex + 1);
        end

        function plotJumpLine(jumps,d,k)
            formatedData = [];
            for i=0:size(jumps,1)/k-1
                temp=[];
                for j = 1:k
                    temp = cat(2,temp,jumps(i*k+j,:));
                end
                formatedData=cat(1,formatedData,temp);
            end
            
            for i=0:k-1
                plot(formatedData(:,i*d+1),formatedData(:,i*d+d));
            end
        end
        
        function drawAnnotations(n,T,acceptedCount,integratedLikelihood)
            str = strcat('Obvs = ', num2str(n));
            str = [str char(10)];
            str = [str strcat('Iterations = ', num2str(T))];
            str = [str char(10)];
%             str = [str strcat('Acpt Ratio = ', num2str((acceptedCount./T)*100), ' %')];
%             str = [str char(10)];
            str = [str strcat('Integrated Likelihood = ', num2str(integratedLikelihood))];
            dim = [0.15 0.05 0.3 0.3];
            annotation('textbox',dim,'String',cellstr(str),'FitBoxToText','on');
            hold on;
        end
        
        function displayPartially()
            [x1,y1,z1] = sphere;
            colx = mean(x1)>=0;
            coly = mean(y1)>=0;
            rowz = mean(z1,2) >=0;
            col = (colx + coly);
            col = find(col == 2);
            surf(x1(rowz,col),y1(rowz,col),z1(rowz,col));
        end
        
        
        % RJMCMC methods
        function x = calculateBk(k,mmax)
            if k == mmax
                x = 0;
            elseif k ==1
                x = 1;
            else
                x = 0.5;
            end
        end
        
        % RJMCMC probabilities
        function x = calculateMergeSpliteProbability(type,randomVector,d,k,kmax,likeliRatio,alpha,g,lambda,delta,indexJ1,indexJ2,dataPoints,muArray,sigmaArray,membershipVector,muJnew,sigmaArrayJnew,featureRelevancy,featureMu,featureSigma)
            % likelihood ratio
            
            disp(['----------------- ' type ' start --------------------']);
            x = log(likeliRatio);
            disp('Likelihood acceptance ratio = ');
            disp(x);
            % p(k+1)./p(k)
            x = x + log(poisspdf(k+1,lambda)./poisspdf(k,lambda));
            disp('line11 acceptance ratio = ');
            disp(x);
            % calculate n1 and n2
            membershipMtx = Utils.calculateMembershipByPoint(d,k+1,dataPoints,muArray,sigmaArray,membershipVector,featureRelevancy,featureMu,featureSigma);
            dataMtxJ1 = dataPoints(membershipMtx(:,indexJ1) == 1,:);
            dataMtxJ2 = dataPoints(membershipMtx(:,indexJ2) == 1,:);
            dataNumJ1 = size(dataMtxJ1,1);
            dataNumJ2 = size(dataMtxJ2,1);
            if dataNumJ1 == 0 && dataNumJ2 == 0
                x = log(0);
                return;
            end
            if x == -Inf || x == Inf
                x == log(rand(1));
            end
            return;  % only use likelihood and prior of k, block rest calculation
            
            
            % (k+1) * Wj1.^(delta-1+L1) * Wj2.^(delta-1+L2)./(Wjnew.^(delta-1+L1+L2)*B(delta,k*delta))
            % computational infeasible?
            %             x = x + log(k+1) + log(membershipVector(indexJ1).^(delta - 1 + dataNumJ1)) + log(membershipVector(indexJ2).^(delta - 1 + dataNumJ2));
            %             disp('line12 acceptance ratio = ');
            %             disp(x);
            %             x = x - log((membershipVector(indexJ1) + membershipVector(indexJ2)).^(delta - 1 + dataNumJ1 + dataNumJ2));
            %             disp('line13 acceptance ratio = ');
            %             disp(x);
            %             x = x - log(beta(delta,k*delta));
            %             disp('line14 acceptance ratio = ');
            %             disp(x);
            % (d-by-d calculation)
            dataMtxJnew = cat(1,dataMtxJ1,dataMtxJ2);
            midpointJnew = mean(dataMtxJnew);
            for j=1:d
                for col=1:2 % lift and right
                    dataMtxJnewPartial = [];
                    sigmaJ1 = sigmaArray((indexJ1-1)*d+1:(indexJ1-1)*d+d,col);
                    sigmaJ2 = sigmaArray((indexJ2-1)*d+1:(indexJ2-1)*d+d,col);
                    if col ==1
                        dataMtxJnewPartial = dataMtxJnew(dataMtxJnew(:,j) <= muJnew(j),:);
                    else
                        dataMtxJnewPartial = dataMtxJnew(dataMtxJnew(:,j) > muJnew(j),:);
                    end
                    if ~isempty(dataMtxJnewPartial)
                        % line 2: sqrt(kappa./2*pi) * exp(-1/2*kappa((MUj1 - XI).^2+(MUj2 - XI).^2-(MUjnew - XI).^2))
                        RJnew = 2 * max(abs(muJnew(j) - dataMtxJnewPartial(:,j)));
                        x = x +log(sqrt(1./(2*pi*RJnew.^2))) - 1./(2*RJnew.^2)*((muArray(indexJ1,j)-midpointJnew(j)).^2 + (muArray(indexJ2,j)-midpointJnew(j)).^2 - (muJnew(j) - midpointJnew(j)).^2 );
                        betaVal = gamrnd(g,200*g/(alpha*RJnew.^2));
                        betaVal = max(0.1,betaVal);
                        %disp('line2 acceptance ratio = ');
                        %disp(x);
                        % line 3
                        x = x + log(betaVal.^alpha) - log(gamma(alpha)) + (-alpha - 1)*log(sigmaJ1(j).^(2) * sigmaJ2(j).^(2)./sigmaArrayJnew(j).^2) - betaVal*(sigmaJ1(j).^(-2) + sigmaJ2(j).^(-2) - sigmaArrayJnew(j).^(-2));
                        %disp('line3 acceptance ratio = ');
                        %disp(x);
                    end
                    % line 5
                    x = x + log((membershipVector(indexJ1) + membershipVector(indexJ2))) + log(abs(muArray(indexJ1,j) - muArray(indexJ2,j))) + log(sigmaJ1(j).^(2) * sigmaJ2(j).^(2));
                    x = x - log(randomVector(2)*(1 - randomVector(2).^2) * randomVector(3) * (1 - randomVector(3))*sigmaArrayJnew(j).^2);
                    %disp('line5 acceptance ratio = ');
                    %disp(x);
                end
            end
            % line 4 (P alloc? probability that this particular allocation is made)
            Dk = 1 - Utils.calculateBk(k+1,kmax);
            Bk = Utils.calculateBk(k,kmax);
            x = x + log (Dk./Bk)-log(betapdf(randomVector(1),2,2))-log(betapdf(randomVector(2),2,2))-log(betapdf(randomVector(3),1,1));
            % special treatment x = x + 200
            % x = x + 200;
            disp('Final acceptance ratio = ');
            disp(x);
            if x == -Inf || x == Inf
                x == log(rand(1));
            end
            disp('----------------- end --------------------');
        end

        function x = calculateBirthDeathProbability(type,d,k,kmax,dataPoints,lambda,delta,weightJnew,muArray,sigmaArray,membershipVector,featureRelevancy,featureMu,featureSigma)
            disp(['----------------- ' type ' start --------------------']);
            % line 1
            %x = log(poisspdf(k+1,lambda)./poisspdf(k,lambda)) - log(beta(k*delta,delta)) + log(weightJnew.^(delta-1)) +log((1-weightJnew).^(size(dataPoints,1)+k*delta-k)) + log(k+1);
            %x = log(poisspdf(k+1,lambda)) - log(poisspdf(k,lambda)) - log(beta(k*delta,delta)) + (delta-1)*log(weightJnew) + (size(dataPoints,1)+k*delta-k)*log(1-weightJnew) + log(k+1);
            x = log(poisspdf(k+1,lambda)) - log(poisspdf(k,lambda)) - log(beta(k*delta,delta)) + (delta-1)*log(weightJnew) + log(k+1);
            
            % line 2
            Dk = 1 - Utils.calculateBk(k+1,kmax)+eps;
            Bk = Utils.calculateBk(k,kmax)+eps;
            membershipMtx = Utils.calculateMembershipByPoint(d,k,dataPoints,muArray,sigmaArray,membershipVector,featureRelevancy,featureMu,featureSigma);
            if isempty(find(sum(membershipMtx,1)==0))
                emptyNum = 0;
            else
                emptyNum = size(find(sum(membershipMtx,1)==0),2);
            end
            x = x + log(Dk) - log((emptyNum + 1)*Bk) + k*log(1-weightJnew) - log(betapdf(weightJnew,1,k));
            disp('Final acceptance ratio = ');
            if x == -Inf || x == Inf
                x = log(rand(1));
            end
            disp(x);
        end
    end
end


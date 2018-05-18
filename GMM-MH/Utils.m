classdef Utils
    %UTILS Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods(Static)
        function x = samplingFromMickey(n,d,delFile,mu,sigma)
            if delFile
                delete 'input.csv';
            end
            if exist('input.csv','file')
                disp('File exists!');
                x=csvread('input.csv');
            else
                if d == 2
                    sigmaArr = [];
                    for i=1:size(mu,1)
                        sigmaArr = cat(3,sigmaArr,eye(size(mu,2))*sigma);
                    end
                    
                elseif d == 3
                    return;
                end
                p = ones(1,size(mu,1))/size(mu,1);
                obj = gmdistribution(mu,sigmaArr,p);
                x = random(obj,n);
                csvwrite('input.csv',x);
            end
        end
        % concatenate membership vector with observations
        function newObvs = calculateMembershipVector(obvs,compMuArray,compSigmaArray,weightArray)
            % initialize membership matrix with observations
            newObvs = cat(2, obvs, zeros(size(obvs,1),size(compMuArray,1)));
            % calculate probability for every observation
            probabilityMtx = [];
            for i=1:size(compMuArray,1)
                probVector = mvnpdf(obvs,compMuArray(i),compSigmaArray(:,:,i));
                probabilityMtx=cat(2,probabilityMtx,probVector);
            end
            % P(x) * P(t-1)
            probabilityMtx = probabilityMtx* diag(weightArray);
            
            for i=1:size(newObvs,1)
                % find out the component that each observation belongs to
                % with the biggest probability
                biggestIndex = -1;
                biggestProb = -1;
                for j=1:size(compMuArray,1)
                    if biggestProb < probabilityMtx(i,j)
                        biggestIndex = j;
                        biggestProb = probabilityMtx(i,j);
                    end
                end
                % generate membership vector for each observation
                newObvs(i,size(obvs,2)+biggestIndex) = 1;
            end
        end
        
        % generate new parameters for mixture model using MH algorithm
        function [newCompMuArray,newCompSigmaArray,newGammaAB,newMuAlphaVector,weight, accepted] = generateParamsMH(obvs,compMuArray,compSigmaArray,weightArray,newWeightArray,gammaAB,zVector, muAlphaVector)
            % generate new parameters from proposal distributions
            % generate sigma array
            
            % generate sigma from Gamma distribution
            newGammaAB = gammaAB;
%             for i=1:size(compMuArray,1)
%                 % A = A_old + Ni
%                 newGammaAB(i,1) = newGammaAB(i,1) + zVector(i);
%                 % B = b + sum(variance)
%                 for j=1:size(obvs,1)
%                     isBelong = obvs(j,size(compMuArray,2) + i);
%                     if isBelong
%                         obv = obvs(j,1:size(compMuArray,2));
%                         obv = obv - compMuArray(i);
%                         powerObv = obv * transpose(obv);
%                         newGammaAB(i,2) = newGammaAB(i,2) + powerObv;
%                     end
%                 end
%             end
            
            newCompSigmaArray=[];
            for i=1:size(compMuArray,1)
                % generate 1xd new sigma vector then convert it as a matrix
                sigmaVector = mvnrnd(sum(compSigmaArray(:,:,i)), compSigmaArray(:,:,i));
                newCompSigmaArray(:,:,i) = abs(diag(sigmaVector));
            end
            
            % generate mus from normal distribution
            newMuAlphaVector = muAlphaVector;
            
            for i=1:size(compMuArray,1)
                % alpha = alpha_old + Ni
                newMuAlphaVector(i,end) = newMuAlphaVector(i,end) + zVector(i);
                % mu = (alpha_old * mu_old + Xi_sum)./alpha
                
                % calculate Xi_avg
                xSum = zeros(1,size(compMuArray,2));
                for j=1:size(obvs,1)
                    isBelong = obvs(j,size(compMuArray,2) + i);
                    if isBelong
                        obv = obvs(j,1:size(compMuArray,2));
                        xSum = xSum + obv;
                    end
                end;
                newMuAlphaVector(i,1:end-1) = (muAlphaVector(i,end)*muAlphaVector(i,1:end-1) + xSum)./newMuAlphaVector(i,end);
            end
            
            % generate new mus
            newCompMuArray = mvnrnd(newMuAlphaVector(i,1:end-1),compSigmaArray);
                        
            % generate acceptance ratio r
            r=1;
            % generate new/old probablilities with new/old mixuture parameters
            fNew = Utils.calculateTargetProb(obvs,newCompMuArray,newCompSigmaArray,newWeightArray);
            fOld = Utils.calculateTargetProb(obvs,compMuArray,compSigmaArray,weightArray);
            % generate f(x|Theta new) / f(x|Theta old)
            r=r*(fNew./fOld);
            
            % generate new/old probabilities of Pi(Theta)
            piNew = Utils.calculatePriorAndPosteriorProb(newCompMuArray,newCompSigmaArray,newGammaAB,newMuAlphaVector);
            piOld = Utils.calculatePriorAndPosteriorProb(compMuArray,compSigmaArray,gammaAB,muAlphaVector);
            % generate Pi(Theta new) / Pi(Theta old)
            r=r*(piNew./piOld);
            
            % generate probabilities of proposal distribution with new/old
            % parameters
            qOldNew=Utils.calculateProposal(compMuArray,compSigmaArray,newCompMuArray,newCompSigmaArray);
            qNewOld=Utils.calculateProposal(newCompMuArray,newCompSigmaArray,compMuArray,compSigmaArray);
            % generate q(Theta old| Theta New) / q(Theta new| Theta old)
            r = r*(qOldNew./qNewOld);
            
            % generate random number from [0,1]
            u = rand(1);
            if r < u
                accepted=0;
                newCompMuArray = compMuArray;
                newCompSigmaArray = compSigmaArray;
                weight = weightArray;
            else
                accepted=1;
                weight = newWeightArray;
            end
        end
        
        % calculate f(x|Theta)
        function prob = calculateTargetProb(obvs,compMuArray,compSigmaArray,weightArray)
            %generate probablilities with mixuture parameters
            obj = gmdistribution(compMuArray,compSigmaArray,weightArray);
            prob = pdf(obj,obvs(:,1:size(compMuArray,2)));
            %prob = prod(prob);% ==0
            % use sum() here instead of prod?
            prob = sum(prob);
        end
        
        % calculate Pi(Theta)
        function prob = calculatePriorAndPosteriorProb(compMuArray,compSigmaArray,gammaAB,muAlphaVector)
            %generate prior probability for new sigma
            sigmaProb = 1;
            for i=1:size(compMuArray,1)
                sigmaProbi = gampdf(sum(compSigmaArray(:,:,i),2),gammaAB(i,1), gammaAB(i,2));
                sigmaProbi = prod(sigmaProbi);
                sigmaProb = sigmaProb*sigmaProbi;
            end           
            % generate prior probability for new mu
            for  i=1:size(compMuArray,1)
                alphai = muAlphaVector(i,end);
                compSigmaArray(:,:,i) = diag(1./(alphai * sum(compSigmaArray(:,:,i))));
            end
            muProb = mvnpdf(compMuArray,muAlphaVector(:,1:end-1),compSigmaArray);   
            % multiply probabilities of k-components together
            muProb = prod(muProb);
            prob=muProb*sigmaProb;
        end
        
        function prob = calculateProposal(compMuArray1,compSigmaArray1,compMuArray2,compSigmaArray2)
            %generate prior probability for new sigma
            sigmaProb = 1;
            for i=1:size(compMuArray1,1)
                sigmaProbTemp = mvnpdf(sum(compSigmaArray1(:,:,i)),compMuArray2(i,:),compSigmaArray2(:,:,i));
                sigmaProb = sigmaProb * sigmaProbTemp;
            end
            
            % generate prior probability for new mu

            muProb = mvnpdf(compMuArray1,compMuArray2,compSigmaArray2);   
            % multiply probabilities of k-components together
            muProb = prod(muProb);
            prob=muProb*sigmaProb;
        end
        
        % a = [vector of positive shape parameters] of Dirichlet
        % distribution
        function y = drchpdf(x,a)
            t1 = gammaln(sum(a))-sum(gammaln(a));
            t2 = sum((repmat(a(1:end-1)-1,size(x,1),1)).*log(x),2);
            t3 = (a(end)-1).*log(1-sum(x,2));
            y = exp(t1 + t2 + t3);
        end
        
        function r = drchrnd(a,n)
            p = length(a);
            r = gamrnd(repmat(a,n,1),1,n,p);
            r = r ./ repmat(sum(r,2),1,p);
        end
        
        function plotDensityRegion(x,mus,sigma,d,k)
            % ploting points without grouping
            gscatter(x(:,1), x(:,2),ones(1,size(x,1)));
            hold on;
            
            % Plot mu points
            for i=1:k
                scatter(mus(i,1), mus(i,2),'*','black');
            end
            for i=1:k
                if d == 3
                    plot_gaussian_ellipsoid(mus(i,:), sigma,2);
                else
                    x = -10:.01:10; %// x axis
                    y = -10:.01:10; %// y axis
                    
                    [X Y] = meshgrid(x,y); %// all combinations of x, y
                    dataMtx = [X(:) Y(:)];
                    Z = mvnpdf(dataMtx,mus(i,:),sigma(:,:,i)); %// compute Gaussian pdf
                    Z = reshape(Z,size(X)); %// put into same size as X, Y
                    contour(X,Y,Z), axis equal  %// contour plot; set same scale for x and y...
                end
                hold on;
            end
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
        
        function drawAnnotations(n,T,acceptedCount)
            str = strcat('Obvs = ', num2str(n));
            str = [str char(10)];
            str = [str strcat('Iterations = ', num2str(T))];
            str = [str char(10)];
            str = [str strcat('Acpt Ratio = ', num2str(acceptedCount./T*100), ' %')];
            dim = [0.15 0.05 0.3 0.3];
            annotation('textbox',dim,'String',cellstr(str),'FitBoxToText','on');
            hold on;
        end
    end
    
end


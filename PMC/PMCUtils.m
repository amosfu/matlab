classdef PMCUtils
    %PMCUTILS Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods(Static)
        % 1.1.2 generate new mus
        function newMuMtx = generateNewMumtx(muMtx,k,d,p)
            newMuMtx = [];
            for i=1:size(muMtx,1)
                tempNewMuMtx = [];
                tempMuMtx = muMtx(i,:);
                v=tempMuMtx(:,end:end);
                for j=0:k-1
                    oldMus = tempMuMtx(:,j*d+1:j*d+d);
                    % generate new mus from normal distribution with old
                    % mus and vk
                    newMus = normrnd(oldMus,v,size(oldMus,1),d);
                    tempNewMuMtx =cat(2,tempNewMuMtx,newMus);
                end
                newMuMtx =cat(1,newMuMtx,cat(2,tempNewMuMtx,tempMuMtx(:,end:end)));
            end
        end
        
        function newMuMtx = calculateWeight(x,oldMuMtx,muMtx,k,d,p,vk)
            newMuMtx = [];
            weightVector = [];
            for i=1:size(muMtx,1)
                tempMuMtx = muMtx(i,:);
                tempoldMuMtx = oldMuMtx(i,:);
                v=tempMuMtx(:,end:end);
                muVector = [];
                oldMuVector = [];
                for j=0:k-1
                    muVector = cat(1,muVector,tempMuMtx(:,j*d+1:j*d+d));
                    oldMuVector = cat(1,oldMuVector,tempoldMuMtx(:,j*d+1:j*d+d));
                end
                % calculate f(x|mu1,mu2)
                obj = gmdistribution(muVector,eye(d),ones(1,k)/k);
                weight = pdf(obj,x(i,:));
                % weightNominator = 1;
                for j=1:k
                    % assume mu1 and mu2 are independent so Pi(mu1, mu2) =
                    % Pi(mu1)*Pi(mu2) (normal distributions with mu = 0 sigma=1)
                    weight = weight* mvnpdf(muVector(j,:));
                    
                    % special operation, make weights bigger to avoid
                    % number loss
                    % weight = weight*10^(8);
                    
                    % --------------------------
                    % Phi(mu1(t)|mu1(t-1),vk)*Phi(mu2(t)|mu2(t-1),vk)
                    % weightNominator = weightNominator *mvnpdf(muVector(j,:),oldMuVector(j,:),eye(d)*v);
                    % ----------------------------
                end
                
                % ---------------------
                % another approach to sum
                % Phi(mu1(t)|mu1(t-1),vk)*Phi(mu2(t)|mu2(t-1),vk) together
                % with all the values of vk
                weightNominator = 0;
                for tempv = vk
                    tempWeightNominator = 1;
                    for j=1:k
                        % select mu1 from t-1 round for the multivarible
                        % Gaussian mixture here?
                        tempWeightNominator = tempWeightNominator *mvnpdf(muVector(j,:),oldMuVector(1,:),eye(d)*tempv);
                    end
                    weightNominator = weightNominator + tempWeightNominator;
                end
                %-----------------------
                
                weightVector =cat(1,weightVector,weight/weightNominator);
            end
            % calculate weight wj with normalization
            weightVector = weightVector/sum(weightVector,1);
            
            newMuMtx=cat(2,muMtx, weightVector);
        end
        
        function [resampledMuMtx, weights] = resamplingByWeight(muMtx,vk)
            % re-sampling mu matrix by weight
            weights = muMtx(:,end:end);
%             disp(weights);
%             disp(sum(weights));
            resampledMuMtx = datasample(muMtx,size(muMtx,1),'Weights',transpose(weights));
            % update weights after re-sampling
            weights=resampledMuMtx(:,end:end);
            resampledMuMtx= resampledMuMtx(:,1:end-1);
            % special treatment, if some vk's is completely removed,
            % recover 5%of total observations for each vk
            vkColumn = resampledMuMtx(:,end:end);
            weights=cat(2,vkColumn,weights);
            survivedVks = unique(vkColumn);
            missingVks = setdiff(vk,survivedVks);
            amount = round(size(vkColumn,1)/20);
            replaceVector = [];
            for missingVk = missingVks
                replaceVector = cat(1,replaceVector,ones(amount,1)*missingVk);
            end
            replaceSize = size(replaceVector,1);
            vkColumn=cat(1,replaceVector,vkColumn(replaceSize+1:end,:));
            resampledMuMtx = cat(2,resampledMuMtx(:,1:end-1),vkColumn);
        end
        
        function x = samplingFromMickey(n,d,delFile)
            if delFile
                delete 'input.csv';
            end
            if exist('input.csv','file') 
                disp('File exists!');
                x=csvread('input.csv');
            else
                if d == 2
                    mu = [-3 3;3 3; 0 -3];
                    sigma = cat(3,[0.5 0;0 0.5],[0.5 0;0 0.5],[1 0;0 1]);
                    p = ones(1,3)/3;
                else
                    % todo: support 3d
                    return;
                end
                obj = gmdistribution(mu,sigma,p);
                x = random(obj,n);
                csvwrite('input.csv',x);
            end
        end
        
        function plotDensityRegion(x,selectedParameters,d,k)
            mus=[]
            for j=0:k-1
                mus = cat(1,mus,selectedParameters(j*d+1:j*d+d));
            end
            sigma = selectedParameters(end-1:end-1);
            weight = selectedParameters(end:end);
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
                    Z = mvnpdf(dataMtx,mus(i,:),eye(d)*sigma); %// compute Gaussian pdf
                    Z = reshape(Z,size(X)); %// put into same size as X, Y
                    contour(X,Y,Z), axis equal  %// contour plot; set same scale for x and y...
                end
                hold on;
            end
        end
        
        function drawAnnotations(MuMtx)
            % count observations related to each variance
            varMtx = PMCUtils.countVariances(MuMtx(:,end-1:end-1));
            str=[];
            for i=1:size(varMtx,1)
                str = [str strcat('V', num2str(i), ' = ', num2str(varMtx(i,1)), ' Obvs = ', num2str(varMtx(i,2)))];
                str = [str char(10)];
            end
            dim = [0.15 0.05 0.3 0.3];        
            annotation('textbox',dim,'String',cellstr(str),'FitBoxToText','on');
            hold on;
        end
        
        function confMtx = countVariances(variances)
            [du,dx,dx]=unique(variances);
            dx=accumarray(dx,1);
            confMtx=[du,dx];
        end
    end
end


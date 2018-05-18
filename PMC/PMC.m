% Population Monte Carlo (PMC) for Gaussian mixture
clear all;
fclose all;
% Change accuracy
digits(500)
% dimension
d=2;
% k-component mixture
k=3;
% Iteration number
T = 10;
% Observation amount
n=1000;
% delete data file before each run
delFile = true;
% variances
vk = [0.01, 0.1, 0.5, 1, 1.5];
p=size(vk,2);

% group amount
m=round(n./p);

% initialize n observations
x = PMCUtils.samplingFromMickey(n,d,delFile);

% initialize mu0 
muMtx=[];
for i=1:p
    muMtx=cat(1,muMtx, cat(2,rand(m,d*k)*3,ones(m,1)*vk(i)));
end

vkWithWeights=[];
newMuMtx=[];
for t=1:T
    if exist('newMuMtx','var') && size(newMuMtx,1) > 0
        muMtx = newMuMtx;
    end
    newMuMtx = PMCUtils.generateNewMumtx(muMtx,k,d,p);
    newMuMtx = PMCUtils.calculateWeight(x,muMtx,newMuMtx,k,d,p,vk);
    [newMuMtx,vkWithWeights] = PMCUtils.resamplingByWeight(newMuMtx,vk);
end
%prepare data for ploting
newMuMtx = cat(2,newMuMtx(:,1:end-1),vkWithWeights);


     
% pick parameters with largest weight to plot density chart
newMuMtx=sortrows(newMuMtx,-(k*d+2));
selectedParameters = newMuMtx(1,:);


% Plot probability density contour
PMCUtils.plotDensityRegion(x,selectedParameters,d,k);

% Draw annotations
PMCUtils.drawAnnotations(newMuMtx);

disp('Finished!');

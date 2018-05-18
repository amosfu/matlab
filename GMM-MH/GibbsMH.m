% Metropolis-within-Gibbs sampling for Gaussian mixture
% clear all;
fclose all;
% Change accuracy
digits(500)

% ------------ static parameters -------------
% dimension
d=2;
% k-component mixture
k=2;
% initial mus and sigma
mu = [5 5;-4 2];
sigma=0.5;
% Gamma distribution initial parameters (all ones)
gammaAB = ones(k,d);
% Normal distribution parameters
muAlphaVector = cat(2,zeros(k,d),ones(k,1)); 

% delete data file before each run
delFile = true;
% Iteration number
T = 100;
% Observation amount
n=1000;
% hyperparameters for prior Dirichlet distribution D(1,...1)
gammaVector = ones(1,k);
% ---------------------------------------------------

% initialize k components, mu, sigma and weights
compMuArray=[];
compSigmaArray = [];
weightArray=ones(1,k)/k;
newWeightArray=[];


for i=1:k
    compMuArray(i,:) = rand(1,d)*-10;
    compSigmaArray= cat(3,compSigmaArray,eye(d));
end
disp('Initialization:');
disp('Mu Array :');
disp(compMuArray);
disp('Sigma Array :');
disp(compSigmaArray);
% initialize n observations
obvs = Utils.samplingFromMickey(n,d,delFile,mu,sigma);

zVector = [];
acceptedCount = 0;
jumps=compMuArray;
for i=1:T
    % generate Zij membership matrix for every observation
    obvs = Utils.calculateMembershipVector(obvs,compMuArray,compSigmaArray,weightArray);
    zVector = sum(obvs(:,d+1:end)); % Nj(t)
    % re-generate weights from Dirichlet distribution
    gammaVector = zVector + gammaVector;
    newWeightArray = Utils.drchrnd(gammaVector,1);
    
    % generate new parameters for mixture model using MH algorithm
    [compMuArray,compSigmaArray,gammaAB,muAlphaVector,weight,accepted]= Utils.generateParamsMH(obvs,compMuArray,compSigmaArray,weightArray,newWeightArray,gammaAB,zVector,muAlphaVector);
    if accepted
        jumps=cat(1,jumps,compMuArray);
        acceptedCount = acceptedCount + 1;
    end
    
    % reformat observations
    obvs = obvs(:,1:d);
    weightArray= weight;
end

% Plot probability density contour
Utils.plotDensityRegion(obvs,compMuArray,compSigmaArray,d,k);
% plot jump traces 
Utils.plotJumpLine(jumps,d,k);
Utils.drawAnnotations(n,T,acceptedCount);

disp('Finished!');
disp('Mu Array :');
disp(compMuArray);
disp('Sigma Array :');
disp(compSigmaArray);
% Gaussian mixture algorithm with EM
clear all;
fclose all;

% Change accuracy
digits(500)

% k: Cluster number
k = 3;

% num: data point number
num = 200;

% dimension
dimension = 2;

% Delete csv file after finish
delFile = false;

% If input data file doesn't exists, generate one and add 200 data points
% into it randomly.

if exist('input.csv','file')
    disp('File exists!');
else
    fid = fopen('input.csv','w');
    for i = 1:num
        if dimension ==3
            fprintf(fid,[num2str(rand(1)) ',' num2str(rand(1)) ',' num2str(rand(1)) '\n']);
        else
            fprintf(fid,[num2str(rand(1)) ',' num2str(rand(1)) '\n']);
        end
    end
    fclose(fid);
end

% Create data point objects
pointMatrix = csvread('input.csv');
dimension = size(pointMatrix,2);

% Initialization
% Generate K Gaussians randomly
components=[];
for i=1:k
    initMean = rand(1,size(pointMatrix,2));
    compoent=GaussianComponent(1./k,initMean ,eye(dimension));
    components=[components compoent];
end
% Insert initial posteriors into data matrix
pointMatrix =[pointMatrix zeros(size(pointMatrix,1),k)];

% (Exit condition?)
n=0;
while n < 100 % 100 iterations
    % Calculate membership probabilities p(k)
    for i =1:size(pointMatrix,1)
        % Calculate g(x;m,sigma)
        for j=1:k
            pointMatrix(i,dimension+j) = Utils.gaussianFunction(pointMatrix(i,1:dimension),components(j),dimension);
        end
        % Calculate denominator of membership probalilities
        membershipArr=[];
        for j=1:k
            membershipArr = [membershipArr (components(j).prior)*pointMatrix(i,dimension+j)];
        end
        
        if sum(membershipArr) == 0
            % Special treatment: because of losing accuracy, if sum = 0, use
            % k-means to allocate data point into one component
            distArr=[];
            for j=1:k
                distance = pointMatrix(i,1:dimension)-components(j).mu;
                distArr = [distArr distance*transpose(distance)];
            end
            minIndex = find(distArr == min(distArr),1);
            % Reset membership probabilities
            for j=1:k
                if j == minIndex
                    pointMatrix(i,dimension+j) = 1;
                else
                    pointMatrix(i,dimension+j) = 0;
                end
            end
        else
            for j=1:k
                 % current posterior ( normalization )
                pointMatrix(i,dimension+j) =membershipArr(j)./sum(membershipArr);
            end
        end
    end
    
    % Re-generate parameters for next iteration (prior, mu and sigma)
    posteriorMatrix = pointMatrix(:,dimension+1:end);
    % Calculate new prior
    posteriorMean = mean(posteriorMatrix);
    for i=1:k
        components(i).prior = posteriorMean(i);
        % Mu & sigma
        posteriorMatrix = pointMatrix(:,dimension+i);
        meanMatrix=[];
        for j=1:size(pointMatrix,1)
            meanMatrix = [meanMatrix; pointMatrix(j,1:dimension)*posteriorMatrix(j)] ;
        end
        % Calculate new mu for component k
        components(i).mu = sum(meanMatrix,1)./sum(posteriorMatrix,1);
        
        for j=1:size(pointMatrix,1)           
            meanDistMatrix (:,:,j)= posteriorMatrix(j)*transpose(pointMatrix(j,1:dimension)-components(i).mu)*(pointMatrix(j,1:dimension)-components(i).mu);
        end
        
        % Calculate new sigma
        components(i).sigma = sum(meanDistMatrix,3)./sum(posteriorMatrix,1);
    end
    n=n+1;
end

% Plotting
grouping = [];
for i=1:size(pointMatrix,1)
    maxIndex = find(pointMatrix(i,dimension+1:end) == max(pointMatrix(i,dimension+1:end)),1);
    grouping = [grouping maxIndex];
end

mus=[];
sigmas=[];

if dimension == 3
    colorstring = 'kbgrymcw';
    clusters = {};
    clusters{k} = [];
    for i=1:size(pointMatrix,1)
        clusters{grouping(i)} = [clusters{grouping(i)}; pointMatrix(i,1:dimension)];
    end
    for i=1:k
        scatter3(clusters{i}(:,1), clusters{i}(:,2),clusters{i}(:,3),'filled',colorstring(rem(i,8)+1));
        hold on;
    end
    % Plot mu points
    for i=1:k
        mus = [mus ; components(i).mu];
        scatter3(mus(:,1), mus(:,2),mus(:,3),'*','black');
    end
else
    gscatter(pointMatrix(:,1), pointMatrix(:,2),grouping);
    hold on;
    
    % Plot mu points
    for i=1:k
        mus = [mus ; components(i).mu];
        scatter(mus(:,1), mus(:,2),'*','black');
    end
end

% Plot probability density contour
Utils.plotDensityRegion(components,dimension);

disp('Finished!');

if delFile
    delete 'input.csv';
end

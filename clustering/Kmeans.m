% K-means algorithm
clear all;
fclose all;

% k: Cluster number
k = 3;

% num: data point number
num = 200;

% dimension
dimension = 3;

% Delete csv file after finish
delFile = true;

% If input data file doesn't exists, generate one and add 200 data points
% into it randomly.

if exist('input.csv','file')
    disp('File exists!');
else
    fid = fopen('input.csv','w');
    for i = 1:num
        if dimension == 3
            fprintf(fid,[num2str(100*rand(1)) ',' num2str(100*rand(1)) ',' num2str(100*rand(1)) '\n']);
        else
            fprintf(fid,[num2str(100*rand(1)) ',' num2str(100*rand(1)) '\n']);
        end
    end
    fclose(fid);
end

% Create data point objects
pointMatrix = csvread('input.csv');
dimension = size(pointMatrix,2);

% Generate centroids randomly
centPoints = 100*rand(k,size(pointMatrix,2));
% Create empty clusters
clusters = {};
clusters{k} = [];

isEqual = false;
while ~isEqual
    % Assign data points into clusters.
    [isEqual,clusters] = Utils.assignPoints(centPoints,clusters,pointMatrix);
    % regenerate centroids
    centPoints = Utils.regenerateCentroids(centPoints,clusters);
end

% determine 2-d or 3-d by input arguments number
colorstring = 'kbgrymcw';
for i = 1:length(clusters)
    selCluster = clusters{i};
    if dimension == 3
        scatter3(selCluster(:,1),selCluster(:,2),selCluster(:,3),'filled',colorstring(rem(i,8)+1));
        hold on;
        scatter3(centPoints(i,1),centPoints(i,2),centPoints(i,3),'*',colorstring(rem(i,8)+1));
    else
        scatter(selCluster(:,1),selCluster(:,2),'filled',colorstring(rem(i,8)+1));
        hold on;
        scatter(centPoints(i,1),centPoints(i,2),'*',colorstring(rem(i,8)+1));
    end
    hold on;
end

if delFile
    delete 'input.csv';
end
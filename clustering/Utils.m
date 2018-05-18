classdef Utils
    %UTILS Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods(Static)
        function [isEqual ,newClusters] = assignPoints(centPoints,clusters, pointMtx)
            centPointsSize = size(centPoints,1);
            pointMtxSize = size(pointMtx,1);
            
            newClusters ={};
            newClusters{centPointsSize} =[];
            %             for i = 1: centPointsSize
            %                 Create empty clusters
            %                 newClusters = [newClusters []];
            %             end
            for i = 1: pointMtxSize
                tempPoint = DataPoint(pointMtx(i,:));
                distArr = [];
                for j = 1:centPointsSize
                    distArr(end +1) = tempPoint.distance(centPoints(j));
                end
                %disp(distArr);
                % Find shortest distance and assign data point into that
                % cluster
                minIndex = find(distArr == min(distArr),1);
                %disp(['Min Index:' num2str(minIndex)]);
                selCluster = newClusters{minIndex};
                % Add data point into cluster
                selCluster = [selCluster;pointMtx(i,:)];
                newClusters{minIndex} = selCluster;
            end
            isEqual = isequal(newClusters,clusters);
            % disp(num2str(length(newClusters)));
        end
        
        
        function newCentPoints = regenerateCentroids(centPoints,clusters)
            for i = 1: length(clusters)
                newCentPoint =  mean(clusters{i});
                centPoints(i,:) =newCentPoint;
            end
            newCentPoints = centPoints;
        end
        
        function probability = gaussianFunction(x,component,d)
            exponent = (x-component.mu)*(inv(component.sigma))*transpose(x-component.mu)/2;
            amplitude = 1./sqrt(((2*pi)^d)*det(component.sigma));
            probability = amplitude*(exp(-exponent));
        end
        
        function plotDensityRegion(components,dimension)
            for i=1:size(components,2)
                if dimension == 3
                    plot_gaussian_ellipsoid(components(i).mu, components(i).sigma,2);
                else
                    x = 0:.001:1; %// x axis
                    y = 0:.001:1; %// y axis
                    
                    [X Y] = meshgrid(x,y); %// all combinations of x, y
                    dataMtx = [X(:) Y(:)];
                    Z = mvnpdf(dataMtx,components(i).mu,components(i).sigma); %// compute Gaussian pdf
                    Z = reshape(Z,size(X)); %// put into same size as X, Y
                    contour(X,Y,Z), axis equal  %// contour plot; set same scale for x and y...
                    %surf(X,Y,Z) %// ... or 3D plot
                end
            end
        end
    end
    
end


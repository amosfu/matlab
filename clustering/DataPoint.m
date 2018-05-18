classdef DataPoint
    %DATAPOINT Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        x = 0;
        y = 0;
        z = 0;
    end
    
    methods
        % Constructor
        function this = DataPoint(args)
            if ~isempty(args)
                this.x = args(1);
            end
            if length(args) >1
                this.y = args(2);
            end
            if length(args) >2
                this.z = args(3);
            end
        end
        
        function dis = distance(this,arr)
            temp = 0;
            if ~isempty(arr)
               temp = temp + (this.x - arr(1))^2; 
            end
            if length(arr) >1
                temp = temp + (this.y - arr(2))^2; 
            end
            if length(arr) >2
                temp = temp + (this.z - arr(3))^2; 
            end
            dis = sqrt(temp);
        end
        
        function arr = toArray(this)
            disp (['(' num2str(this.x) ',' num2str(this.y) ',' num2str(this.z) ')']);
            arr = [this.x this.y this.z];
        end
    end
    
end


classdef GaussianComponent
    
    properties
        prior;
        mu;
        sigma;
    end
    
    methods
        % Constructor
        function this = GaussianComponent(prior,mu,sigma)
            this.prior = prior;
            this.mu = mu;
            this.sigma = sigma;
        end
    end
end



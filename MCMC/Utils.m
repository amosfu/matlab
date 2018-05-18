classdef Utils
    %UTILS Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods(Static)
        function p = mogProb(x, mixWeights, mu, sigma)
            % p(n) = sum_k w(k) N(x(n)|mu(k), sigma(k))
            K = length(mixWeights);
            N = length(x);
            p = zeros(N,1);
            for k=1:K
                p = p + mixWeights(k)*mvnpdf(x(:), mu(k), sigma(k));
            end
        end
        
        function p = target(x, mixWeights, mus, sigmas)
            p = log(Utils.mogProb(x, mixWeights, mus, sigmas));end
        
        function xp = proposal(x, sigma_prop)
            % Nornal Gaussian with mu = 0 and sigma = 10 (initial value)
            xp = x + sigma_prop*randn(1,1);end
        
        function p = proposalProb(x, xprime, sigma_prop)
            p = normpdf(x, xprime, sigma_prop);end
        
        function [samples, naccept] = MetropolisHastings(xinit, Nsamples, targetArgs, proposalArgs, proposalProb)
            % Metropolis-Hastings algorithm
            %
            % Inputs
            % target returns the unnormalized log posterior, called as 'p = exp(target(x, targetArgs{:}))'
            % proposal is a fn, as 'xprime = proposal(x, proposalArgs{:})' where x is a 1xd vector
            % xinit is a 1xd vector specifying the initial state
            % Nsamples - total number of samples to draw
            % targetArgs - cell array passed to target
            % proposalArgs - cell array passed to proposal
            % proposalProb - optional fn, called as 'p = proposalProb(x,xprime, proposalArgs{:})',
            % computes q(xprime|x). Not needed for symmetric proposals (Metropolis algorithm)
            %
            % Outputs
            % samples(s,:) is the s'th sample (of size d)
            % naccept = number of accepted moves
            if nargin < 3, targetArgs = {}; end
            if nargin < 4, proposalArgs = {}; end
            if nargin < 5, proposalProb = []; end
            d = length(xinit);
            samples = zeros(Nsamples, d);
            x = xinit(:)';
            naccept = 0;
            logpOld = Utils.target(x, targetArgs{:});
            t=1
            while t
                xprime = Utils.proposal( x, proposalArgs{:});
                %alpha = Utils.target( xprime, targetArgs{:})/feval(target, x, targetArgs{:});
                logpNew = Utils.target( xprime, targetArgs{:});
                alpha = exp(logpNew - logpOld);
                if ~isempty(proposalProb)
                    qnumer = Utils.proposalProb( x, xprime, proposalArgs{:}); % q(x|x')
                    qdenom = Utils.proposalProb( xprime, x, proposalArgs{:}); % q(x'|x)
                    alpha = alpha * (qnumer/qdenom);
                end
                r = min(1, alpha);
                u = rand(1,1);
                if u < r
                    x = xprime;
                    naccept = naccept + 1;
                    logpOld = logpNew;
                    samples(naccept,:) = x;
                    if naccept>= Nsamples, t=0; end;
                end
            end
        end
        
        function [muAgivenB, sigmaAgivenB] = gaussCondition(mu, Sigma, a, x)
            D = length(mu);
            b = setdiff(1:D, a);
            SAA = Sigma(a,a);
            SAB = Sigma(a,b);
            SBB = Sigma(b,b);
            SBBinv = inv(SBB);
            muAgivenB = mu(a) + SAB*SBBinv*(x(b)-mu(b));
            sigmaAgivenB = SAA - SAB*SBBinv*SAB;
        end
        
        function samples = gibbsGauss(mu, Sigma, xinit, Nsamples)
            % Gibbs sampling for a multivariate Gaussian
            %
            % Input:
            % mu(1:D) is the mean
            % Sigma(1:D, 1:D) is the covariance
            % xinit(1:D) is the initial state
            % Nsamples = number of samples to draw
            %
            % Output:
            % samples(t,:)
            D = length(mu);
            samples = zeros(Nsamples, D);
            x = xinit(:);
            for s=1:Nsamples
                for i=1:D
                    [muAgivenB, sigmaAGivenB] = Utils.gaussCondition(mu, Sigma, i, x);
                    x(i) = normrnd(muAgivenB, sqrt(sigmaAGivenB));
                end
                samples(s,:) = x;
            end
        end
        
        function h = draw_ellipse(x, c, outline_color, fill_color)
            % DRAW_ELLIPSE(x, c, outline_color, fill_color)
            % Draws ellipses at centers x with covariance matrix c.
            % x is a matrix of columns. c is a positive definite matrix.
            % outline_color and fill_color are optional.
            % Written by Tom Minka
            n = 40; % resolution
            radians = [0:(2*pi)/(n-1):2*pi];
            unitC = [sin(radians); cos(radians)];
            r = chol(c);
            if nargin < 3
                outline_color = 'g';
            end
            h = [];
            for i=1:size(x,2)
                y = r*unitC + repmat(x(:, i), 1, n);
                if nargin < 4
                    h = [h line(y(1,:), y(2,:), 'Color', outline_color)];
                else
                    h = [h fill(y(1,:), y(2,:), fill_color, 'EdgeColor', outline_color)];
                end
            end
        end
    end
end

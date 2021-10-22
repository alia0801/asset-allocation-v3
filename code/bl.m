
%v = 3;  % total 3 views
%P = zeros(v, numAssets);
%q = zeros(v, 1);
%Omega = zeros(v);

% View 1
%P(1,1) = 1; 
%q(1) = 0.05;
%Omega(1, 1) = 1e-3;

% View 2
%P(2, 2) = 1; 
%q(2) = 0.03;
%Omega(2, 2) = 1e-3;

% View 3
%P(3, 3) = 1; 
%P(3, 4) = -1; 
%q(3) = 0.05;
%Omega(3, 3) = 1e-5;

%bizyear2bizday = 1/252;
%q = q*bizyear2bizday; 
%Omega = Omega*bizyear2bizday;


%wtsBL = bl_model()

%ax2 = subplot(1,2,1);
%idx_BL = wtsBL>0.001;
%pie(ax2, wtsBL(idx_BL), assetNames(idx_BL));
%title(ax2, portBL.Name ,'Position', [-0.05, 1.6, 0]);
function wtsBL = bl()
T = readtable('close.csv');
assetNames= 1:1:(size(T,2)-1);
retnsT = tick2ret(T(:, 2:end));
benchRetn = retnsT(:, 1);
assetRetns = retnsT(:, assetNames);
numAssets = size(assetRetns, 2);
tau = 0.05;
Sigma = cov(assetRetns.Variables);
C = tau*Sigma;

P = table2array(readtable('matrix_p.csv'));
q = table2array(readtable('matrix_q.csv'));
Omega = table2array(readtable('matrix_omega.csv'));

[wtsMarket, PI] = findMarketPortfolioAndImpliedReturn(assetRetns.Variables, benchRetn.Variables);
mu_bl = (P'*(Omega\P) + inv(C)) \ ( C\PI + P'*(Omega\q));
cov_mu = inv(P'*(Omega\P) + inv(C));
portBL = Portfolio('NumAssets', numAssets, 'lb', 0.05, 'ub',0.5,'budget', 1, 'Name', 'Mean Variance with Black-Litterman');
portBL = setAssetMoments(portBL, mu_bl, Sigma + cov_mu);  
wtsBL = estimateMaxSharpeRatio(portBL);

end


function [wtsMarket, PI] = findMarketPortfolioAndImpliedReturn(assetRetn, benchRetn)
Sigma = cov(assetRetn);
numAssets = size(assetRetn,2);
LB = zeros(1,numAssets);
Aeq = ones(1,numAssets);
Beq = 1;
opts = optimoptions('lsqlin','Algorithm','interior-point', 'Display',"off");
wtsMarket = lsqlin(assetRetn, benchRetn, [], [], Aeq, Beq, LB, [], [], opts);
shpr = mean(benchRetn)/std(benchRetn);
delta = shpr/sqrt(wtsMarket'*Sigma*wtsMarket); 
PI = delta*Sigma*wtsMarket;
end



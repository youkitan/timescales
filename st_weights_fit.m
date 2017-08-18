function [w, Z] = st_weights_fit(B1, B2, TRG)
%
% function [w, e] = st_weights_fit(B1, B2, TRG)
%
% Fit the weights,w, across the sensors to produce a target similarity
% trajectory, TRG, for space-time measures B1 and B2.
%
% INPUT
%
% B1 = spatiotemporal data [Nchan x Nt1]
% B2 = spatiotemporal data [Nchan x Nt2]
% TRG = desired similarity trajectory [Nt x Nt]
%
% OUTPUT
% w = weights across channels producing esimated match to TRG    [Nchan x 1]
% Z = the similarity trajectory matrix arising from the best-fit weights, w
%
 
[Nchan, Nt1] = size(B1);
[Nchan, Nt2] = size(B2);
 
 
B1 = zscore(B1);  %ensure that the columns (spatial pattern at a fixed time) have zero mean and unit variance
B2 = zscore(B2);  %ensure that the columns (spatial pattern at a fixed time) have zero mean and unit variance
 
% weights for diagonalized covariance
 
% problem can be formulated as finding W such that
% B1'*W*B2 = TRG, where W is a diagonal matrix
% [Nt1 X Nchan][Nchan x Nchan][Nchan X Nt2] = [Nt1 X Nt2];
 
% let G = generalized inverse of B2 (with shape Nt2 X Nchan)
% B1'*W*B2*G*B2 = TRG*G2*B2
% B1'*W*B2 = TRG*G*B2
% B1'*W = TRG*G2
 
G = pinv(B2);
Z = TRG*G;
 
% so then we solve a set of Nchan independent regression problems, fitting
% the rows of B1 to the columns of Z
 
w = zeros(Nchan,1);
 
for n = 1:Nchan
   w(n) = regress(Z(:,n), B1(n,:)'); 
end
 
W = diag(w);
 
Z = B1'*W*B2;
 
% Wsquare = pinv(B1')*Gsquare*pinv(B2);
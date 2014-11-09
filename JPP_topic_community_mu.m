function [W, H_top, H_com, M_top, M_com, ObjHistory] = JPP_topic_community_mu(MU,X_top, X_com, R_top, R_com, k, alpha_top, alpha_com, lambda, epsilon, maxiter, verbose)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% X is document x term matrix
% R is the preivous H matrix of previous time step  (init first time with normal NMF H output)
% 
% Optimizes the formulation:
% ||X - W*H||^2 + ||X - W*M*R||^2  + alpha*||M-I||^2 + lambda*[l1 norm Regularization of W and H]
%
% with multiplicative rules.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% fix seed for reproducable experiments
rand('seed', 14111981);

% initilasation
n = size(X_top, 1);
v1 = size(X_top, 2);
v2 = size(X_com,2);
% randomly initialize W, Hu, Hs.
W  = abs(rand(n, k));
H_top = abs(rand(k, v1));
M_top = abs(rand(k,k));
I = speye(k,k);
Ilambda = I*lambda;
H_com = abs(rand(k, v2));
M_com = abs(rand(k,k));

% constants
trXX_top = tr(X_top, X_top);
trXX_com = tr(X_com, X_com);

% iteration counters
itNum = 1;
Obj = 10000000;

prevObj = 2*Obj;

%this improves sparsity, not mandatory.


while((abs(prevObj-Obj) > epsilon) && (itNum <= maxiter)),

     J_top= M_top*R_top;
	 J_com = M_com * R_com;
	 W = W .* ( ( MU*X_top*(H_top'+J_top') + (1-MU)*X_com*(H_com'+J_com')  ) ./ max( W*(MU*(J_top*J_top')+(1-MU)*(J_com*J_com')+MU*(H_top*H_top')+(1-MU)*(H_com*H_com') + lambda) ,eps) );
     %W =  W .* ( X*(H'+J')  ./ max(W*((J*J')+(H*H')+ lambda),eps) );
     WtW =W'*W;
     WtX_com = W'*X_com;
	 WtX_top = W'*X_top;
	 M_top = M_top .* ( (WtX_top*R_top' + alpha_top*I ) ./ max( (WtW*M_top*R_top*R_top') + ( (alpha_top)*M_top )  +lambda,eps) );     
	 M_com = M_com .* ( (WtX_com*R_com' + alpha_com*I ) ./ max( (WtW*M_com*R_com*R_com') + ( (alpha_com)*M_com )  +lambda,eps) );     
     %M = M .* ( ((WtX*R') + (alpha*I)) ./ max( (WtW*M*R*R') + ( (alpha)*M)+lambda,eps) );     
	 H_top = H_top .* (WtX_top ./ max(WtW*H_top+lambda,eps));
	 H_com = H_com .* (WtX_com ./ max(WtW*H_com+lambda,eps));
     %H = H .* (WtX./max(WtW*H+lambda,eps));
     prevObj = Obj;
	 Obj = computeLoss(X_top, X_com, W, H_top, H_com, M_top, M_com, R_top, R_com, lambda, alpha_top, alpha_com, trXX_top, trXX_com, I);
     %Obj = computeLoss(X,W,H,M,R,lambda,alpha, trXX, I);
     delta = abs(prevObj-Obj);
 	 ObjHistory(itNum) = Obj;
 	 if verbose,
            fprintf('It: %d \t Obj: %f \t Delta: %f  \n', itNum, Obj, delta); 
     end
  	 itNum = itNum + 1;
end
function [trAB] = tr(A, B)
	trAB = sum(sum(A.*B));
end    
function Obj = computeLoss(X_top, X_com,W,H_top, H_com,M_top, M_com,R_top, R_com,reg_norm,reg_temp_top, reg_temp_com, trXX_top, trXX_com, I)
    WtW = W' * W;
    MR_top = M_top*R_top;
	MR_com = M_com*R_com;
    WH_top = W * H_top;
	WH_com = W * H_com;
    WMR_top = W * MR_top;    
	WMR_com = W * MR_com;
    tr1_top = trXX_top - 2*tr(X_top,WH_top) + tr(WH_top,WH_top);
    tr1_com = trXX_com - 2*tr(X_com,WH_com) + tr(WH_com,WH_com);
    tr2_com = trXX_com - 2*tr(X_com,WMR_com) + tr(WMR_com,WMR_com);
    tr2_top = trXX_top - 2*tr(X_top,WMR_top) + tr(WMR_top,WMR_top);
    tr3_top = reg_temp_top*(tr(M_top,M_top) - 2*trace(M_top)+ trace(I));
    tr3_com = reg_temp_com*(tr(M_com,M_com) - 2*trace(M_com)+ trace(I));
    tr4_top = reg_norm*(sum(sum(H_top)) + sum(sum(W)) + sum(sum(M_top)) );
    tr4_com = reg_norm*(sum(sum(H_com)) + sum(sum(W)) + sum(sum(M_com)) );
    Obj = MU*(tr1_top+ tr2_top+ tr3_top+ tr4_top);    
    Obj = Obj + (1-MU)*(tr1_com+ tr2_com+ tr3_com+ tr4_com);    
end



end

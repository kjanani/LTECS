%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Copyright (c) 2014 Yahoo! Inc.
%Copyrights licensed under the MIT License. See the accompanying LICENSE file for terms.
%Author: Amin Mantrach  - amantrac at yahoo - inc dot com - http://iridia.ulb.ac.be/~amantrac/
%This is demo file on how to use the JPP decomposition,
%it will produce the final scores in terms of micro F1, macro F1, NAP and NDCG
%the data set used is the TDT2 data set publicaly available from: http://www.nist.gov/speech/tests/tdt/tdt98/index.htm, 
%We are using the matlab version available here: http://www.cad.zju.edu.cn/home/dengcai/Data/TextData.html
%The demo is configured to use 6 topics (k=6, you can change it)
%it initialize the system the first week, using NMF
%then it computes the result the remaining week from 2 to 26.
%intermediary results are displayed at each step for
%JPP, using NMF on the current timestamp (tmodel in the paper), and NMF
%on a fixed starting period timestamp (fix model).
%The demo file use lambda 10000000 this can be changed.
%In case of news, we observed that for a periof of one day, putting high
%value of lambda is the best, as we put emphasis on the past
%If you have a prior on high periodicity of the events, use value =1
%if you don't know, you can do a simple cross-val experimentation 
%a set lambda using a validation set
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%clear all;
%We load the data such that we have 3 matrices
%X doc x words, T doc x time step and Y doc x label


%load TDT2.mat;
X_top=fea_top; % feature for topics
X_com=fea_com; % feature for communities
%load T.mat; 
%Y=[];
%k=6; %We fix the nb of top classes to track
%for v =[1:k]
%    Y  = [Y gnd==v];
%end
% gnd = ground truth of which news belongs to which topic
Y = gnd;

%X = X(find(sum(Y,2)),:); %% JAN: Find which documents are valid for the first 6 topics
%Y = Y(find(sum(Y,2)),:); %% JAN: Same as above
%T = T(find(sum(Y,2)),:);

%load ynewsData;
%[s i ]= sort(sum(Y),'descend');
k=5;
%Y = Y(:,i(1:k));
%X = X(find(sum(Y,2)),:);
%Y = Y(find(sum(Y,2)),:);
%T = T(find(sum(Y,2)),:);



 numlambda=0;


%flag variable
JPPflag=true;
MR = [];
MR_MAP = [];
MRO = [];
MRbaseline =[];
MRfix =[];
MR_JPP = [];
MR_JPP_MAP = [];


regl1nmf = 0.0005;

regl1jpp = 0.05;

epsilon = 0.01;

maxiter = 100;
mu = 1;
for lambda = [10000000]
lambda_top = 10^7;
lambda_com = 10^7;
numlambda = numlambda+1;


%the start time period used for init of W(1) and H(1), using normal NMF
for start= [1]


% Calculate Xt for topic and community (in tfidf form)
t = find(sum(T(:,start),2)); %% Indices of documents active in t=start
Xt_top = X_top(t,:); %% Xt = X matrix for active documents at this time period
idf = log(size(Xt_top,1)./(sum(Xt_top>0)+eps)); 
IDF = spdiags(idf',0,size(idf,2),size(idf,2));
Xtfidf_top = L2_norm_row(Xt_top*IDF); %% Get tfidf of Xt
Xt_com = X_com(t,:);
idf = log(size(Xt_com,1)./(sum(Xt_com>0)+eps));
IDF = spdiags(idf',0,size(idf,2),size(idf,2));
Xtfidf_com = L2_norm_row(Xt_com*IDF);


%call NMF with L1 norm for Xtfidf for topic
[W_top H_top] = NMF(Xtfidf_top, k, regl1nmf, epsilon, maxiter, false);
Hfixmodel = L2_norm_row(H_top);
Hbaseline2= H_top;
HA_top=H_top;
H_JPP = H_top;
[W_com H_com] = NMF(Xtfidf_com, k, regl1nmf, epsilon, maxiter, false);


%number of period we consider
finT = size(T,2); %% Should be the number of weeks we consider


%for all the consecutive periods
for weeks = [start+1:finT]

fprintf('\n=========================\n');
fprintf('day number %i:\n',weeks);
fprintf('----------------\n');
%compute the grountruth as the top 10 words of the center of mass each label set    
nbtopicalwords=10;
t = find(sum(T(:,start:weeks),2));
Xt_top = X_top(t,:);
idf = log(size(Xt_top,1)./(sum(Xt_top>0)+eps)); %% 1
IDF = spdiags(idf',0,size(idf,2),size(idf,2)); %% 2
Xtfidf_top = L2_norm_row(Xt_top*IDF); %% and 3 are calculating the tfidf
Yt = Y(t,:);
Htrue_top = Yt'*Xtfidf_top;
Htrue_top = L2_norm_row(Htrue_top);
[void I]=sort(Htrue_top,2,'descend');

for i=1:size(Htrue_top,1)
      Htrue_top(i,I(i,1:nbtopicalwords))=1;
      Htrue_top(i,I(i,nbtopicalwords+1:end))=0;
end
% Htrue_top is a binary matrix indicating which are the top words in each topic



    
    
    t = find(sum(T(:,[weeks]),2)); % Indices of all the documents active now
    Xt_top = X_top(t,:); 
    idf = log(size(Xt_top,1)./(sum(Xt_top>0)+eps)); %% 1
    IDF = spdiags(idf',0,size(idf,2),size(idf,2)); %% 2
    Xtfidf_top = L2_norm_row(Xt_top*IDF); %% and 3 are calculating the tfidf
    if(size(Xtfidf_top,1)==0)
        continue;
    end
	Xt_com = X_com(t,:); 
    idf = log(size(Xt_com,1)./(sum(Xt_com>0)+eps)); %% 1
    IDF = spdiags(idf',0,size(idf,2),size(idf,2)); %% 2
    Xtfidf_com = L2_norm_row(Xt_com*IDF); %% and 3 are calculating the tfidf
    if(size(Xtfidf_com,1)==0)
        continue;
    end

     
    
 
    
    Ho_top=H_top;
	Ho_com=H_com;
    Ho_JPP = H_JPP;
    
    if(JPPflag)
      fprintf('computing JPP multimodal decomposition...');
      [W, H_top, H_com, M_top, M_com, OBJHIST] = JPP_topic_community_mu(mu,Xtfidf_top, Xtfidf_com, Ho_top, Ho_com, size(Ho_top,1), lambda_top, lambda_com, regl1jpp, epsilon, maxiter, false);
      
      %fprintf('[ok]\ncomputing JPP decomposition...');
      %[W_JPP, H_JPP, M_JPP, OBJHIST] = JPP(Xtfidf_top, Ho_JPP, size(Ho_top,1), lambda, regl1jpp,  epsilon, maxiter, false);
    end
 
%     if(numlambda==1)
%          fprintf('[ok]\ncomputing NMF decomposition for topic...'); 
%         [void Hbaseline2_top] = NMF(Xtfidf_top, k,regl1nmf, epsilon, maxiter, false);
%         Hbaseline_top = L2_norm_row(Hbaseline2_top);          
%         fprintf('[ok]\n');
% 
%        fprintf('computing NMF decomposition for community...'); 
%        [void Hbaseline2_com] = NMF(Xtfidf_com, k,regl1nmf, epsilon, maxiter, false);
%        fprintf('[ok]\n');
%        Hbaseline_com = L2_norm_row(Hbaseline2_com);          
%     end
%    
   
    %JPP baseline
% 	Hev = L2_norm_row(H_JPP);
%     if(JPPflag),
%        Hmax_JPP = [];
%        for i=[1:size(Htrue_top,1)]
%         max = Htrue_top(i,:)*Hev(1,:)';
%         maxi = 1;
%         for j=[2:size(Hev,1)]
%            val =  Htrue_top(i,:)*Hev(j,:)';
%            if (max < val)
%                max = val;
%                maxi = j;
%            end
%         end
%         Hmax_JPP = [Hmax_JPP; Hev(maxi,:)];
%        end
%     end
%    

	% our multimodal method
    Hev = L2_norm_row(H_top);
    if(JPPflag),
        Hmax_top = [];
        for i=[1:size(Htrue_top,1)]
         max = Htrue_top(i,:)*Hev(1,:)';
         maxi = 1;
         for j=[2:size(Hev,1)]
            val =  Htrue_top(i,:)*Hev(j,:)';
            if (max < val)
                max = val;
                maxi = j;
            end
         end
         Hmax_top = [Hmax_top; Hev(maxi,:)];
        end
    end
    
%     if(numlambda==1),[NDCG] = performanceNDCG(Hmax_JPP,Htrue_top);
%                 Hmaxbaseline = [];
%                 for i=[1:size(Htrue_top,1)]
%                  max = Htrue_top(i,:)*Hbaseline_top(1,:)';
%                  maxi = 1;
%                  for j=[2:size(Hbaseline_top,1)]
%                     val =  Htrue_top(i,:)*Hbaseline_top(j,:)';
%                     if (max < val)
%                         max = val;
%                         maxi = j;
%                     end
%                  end
%                  Hmaxbaseline = [Hmaxbaseline; Hbaseline_top(maxi,:)];
%                 end
% 
% 
%                 Hmaxfix = [];
%                 for i=[1:size(Htrue_top,1)]
%                  max = Htrue_top(i,:)*Hfixmodel(1,:)';
%                  maxi = 1;
%                  for j=[2:size(Hfixmodel,1)]
%                     val =  Htrue_top(i,:)*Hfixmodel(j,:)';
%                     if (max < val)
%                         max = val;
%                         maxi = j;
%                     end
%                  end
%                  Hmaxfix = [Hmaxfix; Hfixmodel(maxi,:)];
%                 end
%                                 
%     end
%      R=[];
%     
%      
    
    if(JPPflag),
        [NDCG] = performanceNDCG(Hmax_top,Htrue_top);
        MR = [MR; [NDCG]];
        fprintf('JPP topic community  scores - NDCG: %f\n',NDCG);
    %   [NDCG] = performanceNDCG(Hmax_JPP,Htrue_top);
    %   MR_JPP = [MR_JPP; [NDCG]];
    %   fprintf('JPP unimodal scores - NDCG: %f\n',NDCG);
       
        [NDCG] = performanceMAP(Hmax_top,Htrue_top);
         MR_MAP = [MR_MAP; [NDCG]];
         fprintf('JPP topic community  scores - NDCG: %f\n',NDCG);
    %    [NDCG] = performanceMAP(Hmax_JPP,Htrue_top);
    %    MR_JPP_MAP = [MR_JPP_MAP; [NDCG]];
    %    fprintf('JPP unimodal scores - NDCG: %f\n',NDCG);
       
    end
    
    
    
%     if (numlambda==1),
%   	  Rbaseline = [];
%   	  [NDCG] = performanceNDCG(Hmaxbaseline,Htrue_top);
%  	   MRbaseline = [MRbaseline;[NDCG] ];
%         fprintf('t-model  scores -  NDCG: %f\n',NDCG);
% 
%     
%   	  Rfix = [];
%   	  [NDCG] = performanceNDCG(Hmaxfix,Htrue_top);
%   	  MRfix = [MRfix;  [NDCG]];
%       fprintf('fix-model  scores - NDCG: %f\n',NDCG);
%     end
    fprintf('=========================\n');

end %end for weeks


end %for start 



end %for lambd
mmr = mean(MR);
fprintf('JPP topic community Avg scores NDCG: %f\n',mmr(1));
%mmr = mean(MR_JPP);
%fprintf('JPP unimodal Avg scores NDCG: %f\n',mmr(1));
mmr = mean(MR_MAP);
fprintf('JPP topic community Avg scores MAP: %f\n',mmr(1));
%mmr = mean(MR_JPP_MAP);
%fprintf('JPP unimodal Avg scores MAP: %f\n',mmr(1));


 %mmr = mean(MRbaseline);
 %fprintf('t-model Avg NMF scores NDCG: %f\n',mmr(1));
 %mmr = mean(MRfix);
 %fprintf('fix-model Avg NMF scores NDCG: %f\n',mmr(1));









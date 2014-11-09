function MAP = performanceMAP(P,Y)

	[void, idx] = sort(P,2,'descend');
	no_topics = size(Y,1);
	idx = idx(:,1:10);

	MAP = 0;
	for i = 1:no_topics
		MAP = MAP + mean(cumsum(Y(i,idx(i,:)))./(1:10));
	end
	MAP = MAP/no_topics;

end

function [Y_pred,metrics] = predict(Ws,inputs,cutoff,linear_not_log)
numChunks = size(Ws,2);
% W_dict = containers.Map;
% for i=1:numChunks
%     cur_key = hash_size(Ws{i,1}{1,2});
%     W_dict(cur_key) = Ws{i,1}{1,1};
% end
Y = [];
Y_pred = [];
for i=1:numChunks
    cur_X = inputs{1,i}{1,1}{1,1};
    cur_Y = inputs{1,i}{1,1}{1,2};
    Y = [Y;cur_Y];
    numVariables = size(cur_X,2);
    cur_size = inputs{1,i}{1,2};
    [cur_W,best_size,min_diff] = nearest_pattern_chunk_W(Ws,cur_size);
    W_vec = reshape(cur_W,1,[])';
    if(linear_not_log)
        cur_Y_pred = cur_X*W_vec;
    else
        cur_Y_pred = sigmf1(cur_X*W_vec,[1 0]);
    end
    Y_pred = [Y_pred;cur_Y_pred];
end
metrics.precision = sum(Y_pred > cutoff & Y > cutoff)/sum(Y_pred > cutoff);
metrics.recall = sum(Y_pred > cutoff & Y > cutoff)/sum(Y > cutoff);
metrics.tpr = sum(Y_pred > cutoff & Y > cutoff)/sum(Y > cutoff);
metrics.fpr = sum(Y_pred > cutoff & Y <= cutoff)/sum(Y <= cutoff);
end
% function [size_str] = hash_size(chunk_size)
% size_str = strcat(int2str(chunk_size{1,1}),int2str(chunk_size{1,2}),int2str(chunk_size{1,3}));
% end
function [best_W,best_size,min_diff] = nearest_pattern_chunk_W(Ws,size0)
best_W = Ws{1,1}{1,1};
best_size = Ws{1,1}{1,2};
min_diff = size_diff(Ws{1,1}{1,2},size0);
numChunks = size(Ws,1);
for i=1:numChunks
    cur_size = Ws{i,1}{1,2};
    if size_diff(cur_size,size0) < size_diff(best_size,size0)
        best_W = Ws{i,1}{1,1};
        best_size = cur_size;
        min_diff = size_diff(cur_size,size0);
    end
end
end

function [diff] = size_diff(size1,size2)
diff = 0;
for i=1:size(size1,2)
    diff = diff + sum(abs(size1{1,i}-size2{1,i}));
end
end
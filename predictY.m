function [Y_pred,metrics] = predictY(W,X_te,Y_te,cutoff,bias,linear_not_log)
if ~iscell(X_te)
    numVariables = size(X_te,2);
    W_vec = reshape(W,1,numVariables)';
    if(linear_not_log)
        Y_pred = X_te*W_vec;
    else
        Y_pred = sigmf1(X_te*W_vec,[1 bias]);
    end
elseif(~iscell(W))
    numTasks = size(W,2);
    Y_pred = [];
    for i=1:numTasks
        if(linear_not_log)
            Y_pred = [Y_pred;X_te{1,i}*W(:,i)];
        else
            error('"linear_not_log" should be true');
        end
    end
else
    numLevels = size(X_te,2);
    numObservations = size(X_te{1,1},1);
    Y_preds = zeros(numObservations,numLevels);
    for i=1:numLevels
        Y_preds(:,i)=sigmf1(X_te{1,i}*W{1,i},[1 bias]);
    end
    Y_pred = mean(Y_preds,2);
end
metrics.precision = sum(Y_pred > cutoff & Y_te > cutoff)/sum(Y_pred > cutoff);
metrics.recall = sum(Y_pred > cutoff & Y_te > cutoff)/sum(Y_te > cutoff);
metrics.tpr = sum(Y_pred > cutoff & Y_te > cutoff)/sum(Y_te > cutoff);
metrics.fpr = sum(Y_pred > cutoff & Y_te <= cutoff)/sum(Y_te <= cutoff);
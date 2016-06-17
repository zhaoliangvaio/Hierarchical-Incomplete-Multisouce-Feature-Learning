function example_run
load published_data;
[inputs_tr,sizes,XUZ_std] = training_data_formulate(tr_X,tr_U,tr_Z,tr_Y,tr_chunks,5); % the weights of positive obseravations are 5 times than those of negative ones
[Ws] = interactive_lasso(inputs_tr,sizes,0.5,0.5,0,1);% choose classification using logisitc loss
inputs_te = test_data_formulate(te_X,te_U,te_Z,te_Y,te_chunks,XUZ_std);
% load tmp_intermediate_data;
predict(Ws,inputs_te,1,false);
end
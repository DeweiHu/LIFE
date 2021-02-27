clc, clear, close all

vol = niftiread('E:\\OCTA\\eval\\orig_roi.nii.gz');

%%
opts.sigma = 0.5;
opts.useabsolute = 1;
opts.responsetype = 3;
opts.normalizationtype = 1;

result = oof3response(vol, 1:0.25:4, opts);

niftiwrite(result,'E:\\OCTA\\eval\\orig_roi_oof.nii.gz')
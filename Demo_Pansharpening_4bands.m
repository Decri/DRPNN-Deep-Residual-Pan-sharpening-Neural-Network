%% Before running this demo, you need to compile MatConvNet on your device 
%% to make sure you have the MEX functions "vl_nnconv" and "vl_nnrelu" that 
%% are needed when running 'DRPNN_Matconvnet.m'.
clear;close all;
run vl_setupnn; 
scale = 4;
im_gt = imread('./testdata/QB-MS.tif');im_gt = RSgenerate(im2double(im_gt),0,0);
im_gt = modcrop(im_gt,scale);
[hei,wid,channels] = size(im_gt);
im_pan = imread('./testdata/QB-PAN.tif');im_pan = RSgenerate(im2double(im_pan),0,0);
im_pan = im_pan(1:hei*4,1:wid*4);
im_input = imresize(imresize(im_gt,1/scale,'bicubic'),scale,'bicubic');
im_input(:,:,channels+1) = imresize(im_pan,1/scale,'bicubic');
load('DRPNNfor4bands.mat');
tic;
im_fused = double(DRPNN_Matconvnet(im_input,weight,bias));
toc;
im_fused = min(max(im_fused,0),1);
q2n_DRPNN = q2n(im_gt,im_fused,16,16);
ergas_DRPNN = ERGAS(im_gt,im_h,4);
sam_DRPNN = SAM(im_gt,im_h);
imwrite(uint16(RSgenerate(im_fused(:,:,[3 2 1]),1,1)*65535),'DRPNN-fused.tif');
imwrite(uint16(RSgenerate(im_gt(:,:,[3 2 1]),1,1)*65535),'Ground truth.tif');
fprintf(['Q4 of DRPNN fusion:',num2str(q2n_DRPNN),'\n']); 
fprintf(['ERGAS of DRPNN fusion:',num2str(ergas_DRPNN),'\n']);
fprintf(['SAM of DRPNN fusion:',num2str(sam_DRPNN),'\n']);
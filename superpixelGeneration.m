clc;
clear;
load('data/colorMap_IP.mat');
load('data/predictions_indianPines.mat');
predictions = uint8(predictions);
gt = load('Dataset/Indian_pines_gt.mat');
gt = uint8(gt.indian_pines_gt);

% Displaying the gt map predicted by ensemble model
pred_rgbImage = ind2rgb(predictions, c);
subplot(2,3,1);
imshow(pred_rgbImage);

% Displaying the Original gt map
subplot(2,3,2);
gt_rgbImage = ind2rgb(gt, c);
imshow(gt_rgbImage);

[pred_L,pred_N] = superpixels(pred_rgbImage, 555);
pred_bw = boundarymask(pred_L);
subplot(2,3,4);
imshow(imoverlay(pred_rgbImage, pred_bw, 'white'));
subplot(2,3,5);
[gt_L,gt_N] = superpixels(gt_rgbImage, 150);
gt_bw = boundarymask(gt_L);
imshow(labeloverlay(gt_rgbImage, gt_bw));

max_class = find_max_class(predictions, pred_L);
final_pred = zeros(size(predictions));
for i = 1:size(predictions,1)
    for j = 1:size(predictions,2)
        if(gt(i,j) == 0)
            continue;
        end
        final_pred(i,j) = max_class((pred_L(i,j))+1, 1);
    end
end
final_pred = uint8(final_pred);
final_pred_rgbImage = ind2rgb(final_pred, c);
subplot(2,3,3);
imshow(final_pred_rgbImage);

function result = find_max_class(img, L)
    L = L + 1;
    sp_count = max(max(L));
    class_cnt = max(max(img));
    [rows,cols] = size(img);
    y = zeros(sp_count, class_cnt);
    for j = 1:rows
        for k = 1:cols
            if(img(j,k) == 0)
                continue;
            end
            y(L(j,k), img(j,k)) = y(L(j,k), img(j,k)) + 1;
        end
    end
    [max_ele, max_idx] = max(y,[],2);
    max_idx(max_ele == 0) = 0;
    result = max_idx;
end

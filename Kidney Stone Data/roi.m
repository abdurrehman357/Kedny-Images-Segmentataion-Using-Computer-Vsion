clc
clear all
close all
warning off

% Read the original image
[filename,pathname]=uigetfile('*.*','Pick a MATLAB code file');
filename=strcat(pathname,filename);
originalImage=imread(filename);

% Show the original image
figure;
imshow(originalImage);

% Convert the original image to grayscale
grayscaleImage=rgb2gray(originalImage);

% Show the grayscale image
figure;
imshow(grayscaleImage);

% Threshold the grayscale image
binaryImage=grayscaleImage>20;

% Show the binary image
figure;
imshow(binaryImage);

% Fill the holes in the binary image
filledImage=imfill(binaryImage,'holes');

% Show the filled binary image
figure;
imshow(filledImage);

% Remove small objects from the binary image
filteredImage=bwareaopen(filledImage,1000);

% Show the filtered binary image
figure;
imshow(filteredImage);

% Preprocess the original image using the binary image
preprocessedImage=uint8(double(originalImage).*repmat(filteredImage,[1 1 3]));

% Show the preprocessed image
figure;
imshow(preprocessedImage);

% Adjust the contrast of the preprocessed image
adjustedImage=imadjust(preprocessedImage,[0.3 0.7],[])+50;

% Show the adjusted image
figure;
imshow(adjustedImage);

% Convert the adjusted image to grayscale
grayscaleAdjustedImage=rgb2gray(adjustedImage);

% Show the grayscale adjusted image
figure;
imshow(grayscaleAdjustedImage);

% Apply median filtering to the grayscale adjusted image
medianFilteredImage=medfilt2(grayscaleAdjustedImage,[5 5]);

% Show the median filtered image
figure;
imshow(medianFilteredImage);

% Threshold the median filtered image
thresholdedImage=medianFilteredImage>250;

% Show the thresholded image
figure;
imshow(thresholdedImage);

% Define the region of interest
[row, col]=size(thresholdedImage);
x1=row/2;
y1=col/3;
ROIx=[x1 x1+200 x1+200 x1];
ROIy=[y1 y1 y1+40 y1+40];
BW=roipoly(thresholdedImage,ROIx,ROIy);

% Show the region of interest
figure;
imshow(BW);

% Apply the region of interest to the thresholded image
segmentedImage=thresholdedImage.*double(BW);

% Show the segmented image
figure;
imshow(segmentedImage);

% Remove small objects from the segmented image
filteredSegmentedImage=bwareaopen(segmentedImage,4);

% Label the connected components in the filtered segmented image
[labels, numLabels]=bwlabel(filteredSegmentedImage);

% Check if a stone is detected in the image
if(numLabels>=1)
    disp('Stone is detected');
else
    disp('No stone is detected');
end

% Generate the ground truth image
% NOTE: Replace 'ground_truth_image.jpg' with the name of the ground truth image file
[filename, pathname] = uigetfile('*.*', 'Pick a MATLAB code file');
filename = strcat(pathname, filename);
groundTruthImage = imread(filename);
groundTruthImage=im2bw(groundTruthImage);

% Resize the ground truth image to match the size of the segmented image
groundTruthImage=imresize(groundTruthImage, size(segmentedImage));

% Calculate the true positives, false positives, false negatives, and true negatives
TP=sum(sum(segmentedImage & groundTruthImage));
FP=sum(sum(segmentedImage & ~groundTruthImage));
FN=sum(sum(~segmentedImage & groundTruthImage));
TN=sum(sum(~segmentedImage & ~groundTruthImage));

% Calculate the accuracy, precision, recall, F
accuracy = (TP+TN)/(TP+FP+FN+TN);
precision = TP/(TP+FP);
recall = TP/(TP+FN);
F = 2*(precision*recall)/(precision+recall);

% Calculate the mean squared error (MSE)
segmentedImage = im2uint8(segmentedImage);
groundTruthImage = im2uint8(groundTruthImage);

MSE = immse(segmentedImage, groundTruthImage);
PSNR = psnr(segmentedImage, groundTruthImage);
elapsedTime = toc;


% Display the evaluation metrics
fprintf('Evaluation metrics for thresholding-based segmentation:\n');
fprintf('Accuracy: %f\n', accuracy);
fprintf('Precision: %f\n', precision);
fprintf('Recall: %f\n', recall);
fprintf('F-measure: %f\n', F);
fprintf('Mean Squared Error: %f\n', MSE);
fprintf('Peak Signal-to-Noise Ratio: %f\n', PSNR);
fprintf('Elapsed Time: %f seconds\n', elapsedTime);


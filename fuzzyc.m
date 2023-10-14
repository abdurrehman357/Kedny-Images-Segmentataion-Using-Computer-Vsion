Irgb = imread('D120313011_ulta (4).jpg');
Igray = rgb2gray(Irgb);

figure
image(Igray,'CDataMapping','scaled')
colormap('gray')
title('Input Image in Grayscale')
I = im2double(Igray);
Gx = [-1 1];
Gy = Gx';
Ix = conv2(I,Gx,'same');
Iy = conv2(I,Gy,'same');
figure
image(Ix,'CDataMapping','scaled')
colormap('gray')
title('Ix')
figure
image(Iy,'CDataMapping','scaled')
colormap('gray')
title('Iy')
edgeFIS = mamfis('Name','edgeDetection');
edgeFIS = addInput(edgeFIS,[-1 1],'Name','Ix');
edgeFIS = addInput(edgeFIS,[-1 1],'Name','Iy');
sx = 0.1;
sy = 0.1;
edgeFIS = addMF(edgeFIS,'Ix','gaussmf',[sx 0],'Name','zero');
edgeFIS = addMF(edgeFIS,'Iy','gaussmf',[sy 0],'Name','zero');
edgeFIS = addOutput(edgeFIS,[0 1],'Name','Iout');
wa = 0.1;
wb = 1;
wc = 1;
ba = 0;
bb = 0;
bc = 0.7;
edgeFIS = addMF(edgeFIS,'Iout','trimf',[wa wb wc],'Name','white');
edgeFIS = addMF(edgeFIS,'Iout','trimf',[ba bb bc],'Name','black');
figure
subplot(2,2,1)
plotmf(edgeFIS,'input',1)
title('Ix')
subplot(2,2,2)
plotmf(edgeFIS,'input',2)
title('Iy')
subplot(2,2,[3 4])
plotmf(edgeFIS,'output',1)
title('Iout')
r1 = "If Ix is zero and Iy is zero then Iout is white";
r2 = "If Ix is not zero or Iy is not zero then Iout is black";
edgeFIS = addRule(edgeFIS,[r1 r2]);
edgeFIS.Rules
Ieval = zeros(size(I));
for ii = 1:size(I,1)
    Ieval(ii,:) = evalfis(edgeFIS,[(Ix(ii,:));(Iy(ii,:))]');
end
figure
image(I,'CDataMapping','scaled')
colormap('gray')
title('Original Grayscale Image')
figure
image(Ieval,'CDataMapping','scaled')
colormap('gray')
title('Edge Detection Using Fuzzy Logic')


% Generate the ground truth image
% NOTE: Replace 'ground_truth_image.jpg' with the name of the ground truth image file
[filename, pathname] = uigetfile('*.*', 'Pick a MATLAB code file');
filename = strcat(pathname, filename);
groundTruthImage = imread(filename);
groundTruthImage=im2bw(groundTruthImage);

% Resize the ground truth image to match the size of the segmented image
groundTruthImage=imresize(groundTruthImage, size(Ieval));

% Calculate the true positives, false positives, false negatives, and true negatives
TP=sum(sum(Ieval & groundTruthImage));
FP=sum(sum(Ieval & ~groundTruthImage));
FN=sum(sum(~Ieval & groundTruthImage));
TN=sum(sum(~Ieval& ~groundTruthImage));

% Calculate the accuracy, precision, recall, F
accuracy = (TP+TN)/(TP+FP+FN+TN);
precision = TP/(TP+FP);
recall = TP/(TP+FN);
F = 2*(precision*recall)/(precision+recall);


% Display the evaluation metrics
fprintf('Evaluation metrics for thresholding-based segmentation:\n');
fprintf('Accuracy: %f\n', accuracy);
fprintf('Precision: %f\n', precision);
fprintf('Recall: %f\n', recall);
fprintf('F-measure: %f\n', F);

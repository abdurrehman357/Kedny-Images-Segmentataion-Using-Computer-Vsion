clc
clear all
close all
warning off

image=imread('D120313011_ulta (4).png');     %Read the image
b=rgb2gray(image);
figure;
imshow(b);
image1=double(image);         %convert into double
image11=image1(:);            %convert the image matrix into vector.
copy=image1;                  %copy the image
[R,C]=size(image1);           %to get the dimension of the Matrix
%make the Centroid
mu=[0 240 120];                %innitialization of the Matrix
%Recalculation of the centroidass
ass1=zeros(R,C);              %zeros matrix for the 1st clusterass
ass2=zeros(R,C);              %zeros matrix for the 2nd clusterass
ass3=zeros(R,C);              %zeros matrix for the 3rd cluster
while(true)
oldmu=mu;                 %check the centroid
for i=1:R               %row
for j=1:C               %coulomn
r=image1(i,j);          %pixel value of the (i,j)co-ordinate
ab=abs((image1(i,j))-mu);%to find out that the pixel will belongs to which cluster

mn=find(ab==min(ab));

if mn(1)==1
ass1(i,j)=r;            %assingning to the 1st clusterend
end
if mn(1)==2
ass2(i,j)=r;            %assingning to the 2nd clusterend
end
if mn(1)==3
ass3(i,j)=r;   %assingning to the 3rd clusterendendend
end
end
end

co1=ass1(:);           %transfer into vector
su1=sum(co1);          %sum of the vector
fi1=find(co1);         %to find non zero elemen
len1=length(fi1);      %to find the length of the non zeor element.
mm1=su1/len1;
mm11=round(mm1);       %new center element.%now to calculate the 2nd element of the centroid

co2=ass2(:);
su2=sum(co2);
fi2=find(co2);
len2=length(fi2);
mm2=su2/len2;
mm22=round(mm2);%now to calculate the 3rd elecment of the centroid.
co3=ass3(:);
su3=sum(co3);
fi3=find(co3);
len3=length(fi3);
mm3=su3/len3;
mm33=round(mm3);
%new centroid
mu=[mm11 mm22 mm33];
if(mu==oldmu)     
break
end
end
%labelling of the clusters
for i=1:R
for j=1:C
if ass1(i,j)>0
ass1(i,j)=1;
end
if ass2(i,j)>0
ass2(i,j)=2;
end
if ass3(i,j)>0
ass3(i,j)=3;
end
end
end
%representing the culustered image
finlcluste=(ass1+ass2+ass3);          %sum up the three labelled cluster
finlcluste1=label2rgb(finlcluste);    %final segmented image%
ff=rgb2gray(finlcluste1);
fff=im2bw(ff);
figure(1);
imshow(image)
title('1.jpg');

figure(2);
imshow(finlcluste1);
title('Clustered Image');

figure(3);
imshow(fff);
title('Clustered Image In Black and White');

m1=rgb2gray(finlcluste1);
%m1=im2bw(m1);

x1=edge(m1,'canny');
figure(3)
imshow(x1);
tic;




%segmentedImage = imread(finlcluste1);
segmentedImage=fff;

% Read the ground truth image
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

% Calculate the evaluation metrics
accuracy = (TP+TN)/(TP+FP+FN+TN);
precision = TP/(TP+FP);
recall = TP/(TP+FN);
F = 2*(precision*recall)/(precision+recall);

segmentedImage = im2uint8(segmentedImage);
groundTruthImage = im2uint8(groundTruthImage);

MSE = immse(segmentedImage, groundTruthImage);
PSNR = psnr(segmentedImage, groundTruthImage);
elapsedTime = toc;

% Display the evaluation metrics
fprintf('Evaluation metrics for K-Means segmentation:\n');
fprintf('Accuracy: %f\n', accuracy);
fprintf('Precision: %f\n', precision);
fprintf('Recall: %f\n', recall);
fprintf('F-measure: %f\n', F);
fprintf('Mean Squared Error: %f\n', MSE);
fprintf('Peak Signal-to-Noise Ratio: %f\n', PSNR);
fprintf('Elapsed Time: %f seconds\n', elapsedTime);








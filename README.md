# polyp_segmentation_senior_project

• This was my final year project. The aim of this project was to build a deep learning model that was able to
perform a binary segmention on colonoscopy images to isolate polyps from the rest of the colon.

• In this project I used the CVC-ClinicDB dataset from Kaggle. As architect I used the UNET architecture which
is fully convolutional neural network. The model takes as input an image and output a probability for each
pixel. Then we apply a threshold to separate the pixels corresponding to our segmented object from those
corresponding to the background. That way we create our mask image. I achieved a meanIoU of 84%

![Figure_1](https://github.com/adoumadje/polyp_segmentation_senior_project/assets/58952237/3a20decd-af4e-45b7-bcbf-fb327e785f7d)
Look at my presentation here:



https://github.com/adoumadje/polyp_segmentation_senior_project/assets/58952237/a41f0d51-0ec6-487b-a160-693d06a34a61


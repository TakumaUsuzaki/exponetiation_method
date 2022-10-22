# exponetiation_method

Exponentiation method is an augmentation method for deep learning. In ecponentiation method, pixel values in an image is exponentiated. This process leads to contrast accentuation. Some other augmentaion methods imitate the imagination process of human, on the other hand, exponentiation method constructs "un-imaginable" image. In this repository, we release the code implementes exponentiation method by PyTorch. 

## Image example

Figure shows lung nodules on CT image which are applied exponentiation method by changing the exponent value 1.0-6.0.

<img src="画像のURL" width="画像のサイズ">

## How to run the test code

Prepare the 4 directories which are composed of "training images categorized as 1", "training images categorized as 2", "test images categorized as 1", and "test images categorized as 2". In our implementation we used ".npy" file to increase execution speed. When you execute "__main__.py", you are required 8 inputs: exponent, epoch, batch size, path for training images categorized as 1, path for training images categorized as 2, path for test images categorized as 1, path for test images categorized as 2, and path for output. In the test code execution, please input as follows.

```
Please Enter an exponent value: 5.0
Please Enter the epoch: 100
Please Enter the batch size: 20
Please Enter Path of category 1 for train: ../tests/train1
Please Enter Path of category 2 for train: ../tests/train2
Please Enter Path of category 1 for test: ../tests/test1
Please Enter Path of category 2 for test: ../tests/test2
Please Enter Output path: ../tests
```

After exesution, you obtain the output json file.

## Citation
When you use code in this repository, please cite papers below.

[Exponentiating pixel values for data augmentation to improve deep learning image classification in chest X-rays
Takuma Usuzaki, Kengo Takahashi, Daiki Shimokawa, Kiichi Shibuya
bioRxiv 2021.03.11.434925; doi: https://doi.org/10.1101/2021.03.11.434925](https://www.biorxiv.org/content/10.1101/2021.03.11.434925v1.abstract)

[Augmentation method for convolutional neural network that improves prediction performance in the task of classifying primary lung cancer and lung metastasis using CT images
Usuzaki, Takuma et al.
Lung Cancer, Volume 160, 175 - 178, doi: https://doi.org/10.1016/j.lungcan.2021.06.021](https://www.lungcancerjournal.info/article/S0169-5002(21)00468-2/fulltext)

## Acknowledgement
The authors acknowledge the National Cancer Institute and the Foundation for the National Institutes of Health, and their critical role in the creation of the free publicly available LIDC/IDRI database used in this study. Data used in this research were obtained from The Cancer Imaging Archive (TCIA) sponsored by the Cancer Imaging Program, DCTD/NCI/NIH, https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI.





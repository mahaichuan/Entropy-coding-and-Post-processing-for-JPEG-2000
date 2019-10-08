# Entropy-coding-and-Post-processing-for-JPEG-2000
This project provides source code of two techniques to improve the compression performance of JPEG-2000, namely entropy coding and post-processing, which was proposed in https://ieeexplore.ieee.org/document/8803835.

Please note that only the code of the neural network model are given, to use this, you need to add I/O operation, both in training and test phase.

The code in folder of bitstream_recompress, is used to compress the wavelet coefficients in two steps: 1. compress the magnitude; 2. compress the sign. The wavelet coefficients are extracted before the inverse-transform of JPEG-2000.

The code in folder of post_processing, is used to post process the output image after JPEG-2000. And we give some visual results in the folder of visual test. The common test dataset Kodak, and the reference software of JPEG-2000, Jasper, are used.

Please refer to the paper for more information, or contact us. Thanks!

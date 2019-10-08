# Entropy-coding-and-Post-processing-for-JPEG-2000
This project provides source code of two techniques to improve the compression performance of JPEG-2000, namely entropy coding and post-processing, which was proposed in https://ieeexplore.ieee.org/document/8803835.

Please note that only the codes of the model are given, to use this, you need to add I/O operation, both in training and test phase.

This code compress the wavelet coefficients in two steps: 1. compress the magnitude; 2. compress the sign. The wavelet coefficients are extracted before the inverse-transform of JPEG-2000.


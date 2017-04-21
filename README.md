# Kaggle Data Science Bowl 
My Work on Kaggle Data Science Bowl as a member of the University of Toronto Data Science Team

I learned how powerful multiprocessing was, especially on AWS instances that had multi-core CPUs (up to 36 for c4.8xlarge!)

## Preprocessing Tests

For smaller dataset `sample_images`, here are the times it took to preprocess all 20 sample patients.

* 2 processes on 4 CPU c4.xlarge: 6m 47s
* 4 processes on 4 CPU c4.xlarge: 4m 29s
* 6 processes on 4 CPU c4.xlarge: 3m 40s
* 8 processes on 8 CPU c4.2xlarge: 2m 09s

For the first 50 images (by path name) in `stage_1` full dataset:

* 10 processes on 8CPU c4.2xlarge: 4m 45s
* 14 processes on 8CPU c4.2xlarge: 4m 14s 

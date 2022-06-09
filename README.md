# Super-Focus: Domain Adaptation for Embryo Imaging via Self-Supervised Focal Plane Regression

To run, first install the dependencies in `requirements.txt`.

Make sure your dataset is in the following layout: `/dataset root/[CLINIC NAME]/[EMBRYO ID]/[FOCAL PLANE].jpg`.

Next, train the autoencoder for the perceptual loss by running `python train_perceptual_ae.py`. Beware: all paths have been replaced with `PLACEHOLDER` - you will need to update them first. Also, we use Weights and Biases to monitor our experiments. Be sure to keep an eye on them.

Once the autoencoder has been trained, you can train the generator networks by running `python train_gan.py`. Again, be sure to replace all the placeholders. The `datasets` folder contains PyTorch datasets for intermediate plane prediction (`t4missingplanes.py`) and upper/lower plane prediction (`t4extrapolation.py`) which should be exchanged in `train_gan.py` as appropriate.

Once you have your trained models (or for validation purposes while you're training models), you can make predictions using `predict.py`. Code for experiments can be seen in `miccai.ipynb`.
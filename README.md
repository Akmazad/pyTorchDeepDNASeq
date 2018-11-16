# pyTorchDeepDNASeq
DNA-seq analysis with deep learning using PyTorch.

This is a PyTorch version of [Akmazad/DeepDNAseq](https://github.com/Akmazad/DeepDNAseq), with some minor changes.

## Usage

To run on my personal computer:

1. SSH into cameronstewart.xyz .
2. Clone the repository.
3. Type the following to run:
   ```bash
   $ python3 main.py
   ```

To run on Raijin:

1. SSH into raijin.nci.org.au .
2. Clone the repository.
3. Type the following to submit the job:
   ```bash
   $ qsub raijinJob.pbs
   ```

Take note from the job script that modules "python3/3.6.2" and "pytorch" must first be loaded.

## PyTorch vs TensorFlow vs Keras

I am suggesting we use PyTorch for the project.  For anyone who has had experience with NumPy, PyTorch will likely be easier to learn and much more intuitive than TensorFlow or Keras.  Having now had experience with all three, I can say I find PyTorch to be the easiest to work with and the most "Pythonic".  PyTorch uses dynamic computational graphs which allow all the features of the Python language to be utilised during the training the testing of a model.  This means conditional statements and loops can be performed using standard Python "if...elif...else", "for", and "while" statements, and it is possible to print out tensor values simply with the Python print() function.  This can significantly ease model-building and debugging, and is far more intuitive than having to utilise something like the tf.cond() function in TensorFlow.  Note that TensorFlow now offers "eager execution" mode, which also allows TensorFlow/Keras to utilise dynamic computational graphs, but my preference is still for PyTorch.

From what I have seen and heard, PyTorch appears to be a bit faster than TensorFlow, which is faster than Keras (with TensorFlow as the backend).  Keras is great for simple models, but would likely require some of the code to be written in TensorFlow for more complex/unique models.  PyTorch and TensorFlow both offer a greater deal of flexibility than Keras.

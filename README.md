# GPianoroll

Music is a mix of art and science: a way for humans to express themselves, but also a field with heavy maths and physics involved! A musical piece's impression depends on the physical properties of the sounds that make it, but also on the listener they reach out to. In this project, we propose an evaluation method for Deep Learning generative models that leverages human feedback as our quality measurement for music. We perform Bayesian Optimization over a low-dimensional latent space where we take samples from, map these samples to the high-dimensional latent space of the musical score generator MuseGAN, and ask the final user to evaluate how they like the output of the model. After some BO iterations, our method finds which point of the latent space is more to the subject's liking, and extracts one final sample.

Want to see some of our samples? Check out our [GitHub pages](https://mikceroese.github.io/GPianoroll) site

# Files

Our work includes the following files:
 - **A notebook for training our baseline**, which is a custom version of the **MuseGAN** model, built with PyTorch.
   `MuseGAN_DBG_chroma.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mikceroese/GPianoroll/blob/1ab74a52e59d844751b2a68ec026f3ab4e1aba22/MuseGAN_DBG_chroma.ipynb)
   The notebook is completely runnable in Colab, and we optimized RAM and GPU usage for fully online reproducibility!
 - **A pretrained version** of the baseline, result of running the notebook.
 - **Code for running experiments**, which include the description of the model, a GUI, and auxiliary code for loading the model and running the experiments. 
 - **Example MIDI files**, available in the 'docs' folder.


# Running the code

In order to run the code you need to:
- Install the necessary dependencies. For the notebook, dependencies are included in the first cell.
  Most importantly, you must install PyTorch, pypianoroll, PrettyMIDI and BayesOpt locally to run the optimization.
- Select a trained model. You can use the provided checkpoint or train another one using the provided notebook.
  In case you decide to use the notebook, it contains instructions for generating the model and saving it.
- Select a soundfont in order to play the MIDI files using the built in interface. There are plenty of
  online sources for free-use .sf2 files.
- Make sure the code in gpianoroll.py matches your configuration (future work may include command line options for this)

After everything is in place, you can run the optimization process by executing gpianoroll.py in your interpreter.


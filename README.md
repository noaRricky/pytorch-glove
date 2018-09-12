# pytorch-glove

## Overview

This is an implementation of [GloVe](http://nlp.stanford.edu/projects/glove/) (Global Vectors for Word Representation), a model combine the glov matrix factorizaton methods and local context window method for learning word vectors. The model was originally developed with C by Jeffery Pennington, Richard Socher, and Christopher Manning.

This is [pytorch](https://pytorch.org/) version of GloVe. "PyTorch is a deep learning framework for fast, flexible experimentation."

## Credit

Thanks for Jeffery Pennington, Richard Socher, and Christopher Manning, who developed the model, published a paper about it, and released an C implementation version of the model.

I also appreciate Stanford NLP team upload course CS224n (Natural Language Processing) resource online. This is really a exciting experience for a Chinese student like me to have chance to study online courses offered by Standford.

Thanks also to [GradySimon](https://github.com/GradySimon), who wrote a Tensorflow implementation of the model and [hans](https://github.com/hans) post a blog describing the implementation detail.

## References

- Pennington, J., Socher, R., & Manning, C. D. (2014). [Glove: Global vectors for word representation](http://nlp.stanford.edu/pubs/glove.pdf). Proceedings of the Empiricial Methods in Natural Language Processing (EMNLP 2014), 12, 1532-1543.
- [stanfordnlp/GloVe](https://github.com/stanfordnlp/GloVe) - Pennington, Socher, and Manning's C implementation of the model
- [hans/glove.py](https://github.com/hans/glove.py) - Jon Gauthier's Python implementation
- [A GloVe implementation in Python](http://www.foldl.me/2014/glove-python/) - Jon's blog post on the topic
- [GradySimon/tensorflow-glove](https://github.com/GradySimon/tensorflow-glove) implement GloVe by using Tensorflow
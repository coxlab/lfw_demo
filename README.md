Demo: using L3 models on LFW
============================

Simple high-level driver code to extract features.

Assuming that you have already installed standard dependencies (`numpy`,
`scipy`, `sklearn`, `skimage`, `Theano`, etc.) you'll just need to
install the non-standard ones (from git submodules):

    $ git submodule update --init
    $ ./bootstrap.sh


To run the driver code:

    $ python demo.py
    (...)
    average accuracy = 0.803833333333
    time = 1012.0362260

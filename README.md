# python-libartificial

This is the python wrapper for [libartificial](https://github.com/fetacore/libartificial).

It is CPU ([OpenBLAS](https://github.com/xianyi/OpenBLAS)) implemented but soon with support for cuBLAS if I get my hands on an NVIDIA GPU. I have plans to extend it for CNNs and RNNs.

The feedforward procedure does not have a hardcoded depth (it can have as many layers as you want).

## Getting Started

The library is created with Linux machines in mind but OSX users should not have a problem if they have gcc installed.
To compile the library with Visual Studio you need to do a whole lot of processing to build OpenBLAS yourself and include the pthreads.dll in your build system but I do not recommend it.

In order to get and build libartificial you have to do the following (assuming working installation of git)

```
git clone https://github.com/jroukis/libartificial.git
cd libartificial
rm -rf .git
make cpu
mv libartificial.* ../
cd ../
rm -rf libartificial
```

After that get the python wrapper

```
git clone https://github.com/jroukis/python-libartificial.git
cd python-libartificial
rm -rf .git
mv ../libartificial.* libartificial/shared
```

Afterwards, create a python virtual environment and put the whole folder "libartificial" inside <virtualenv_folder>/lib/python3.6/site-packages
and you are done!

### Prerequisites

In order to compile the library for CPU you need to install [OpenBLAS](https://github.com/xianyi/OpenBLAS).

## Donations

If you like my work and/or you want to use it for your own projects or want me to create a custom recipe for you, I would gladly accept your donations at:

BTC: 1HzxXZPQSNg7U53XoBSWCpugKUg5DaZELu

ETH: 0xf09fce52f7ecd940cae2826deae151b6495354f6

## License

Copyright (c) Jim Karoukis.
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

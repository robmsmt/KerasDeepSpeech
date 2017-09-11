# Keras DeepSpeech
[![Build Status](https://travis-ci.org/mlrobsmt/KerasDeepSpeech.svg?branch=master)](https://travis-ci.org/mlrobsmt/KerasDeepSpeech)
<!-- [![Build Status](https://travis-ci.com/rmsmith88/darkspeech.svg?token=y6aR2NnYkpGbbYKLNpwK&branch=master)](https://travis-ci.com/rmsmith88/darkspeech) -->


<!-- ___ -->

Repository for experimenting with different CTC based model designs for ASR.

## Overview

<!-- ![Overview kDS](https://raw.githubusercontent.com/mlrobsmt/KerasDeepSpeech/master/preproc/overview.gif "Overview of kDS and batchgen") -->
<img src="https://raw.githubusercontent.com/mlrobsmt/KerasDeepSpeech/master/preproc/overview.gif" align="left" height="279" width="340">

<!-- ## Existing Architectures - model.py -->
<!-- 1. Arch 0 - DS1 (3FC+BLSTM+SOFTMAX) with dropout -->
<!-- 2. Arch 1 - DS1 (3FC+BLSTM+SOFTMAX) dropout -->
<!-- 3. Arch 2 - DS2 (1D conv+BGRU+FC+SOFTMAX) -->
<!-- 4. Arch 3 - own FC+ -->
<!-- 5. Arch 4 - Graves2006 (conv) -->
<!-- 6. Arch 2 - DS2 (conv) -->


## Setup
1. Recommended > use virtualenv installed with python2.7 (or 3.x untested)
2. `git clone https://github.com/mlrobsmt/KerasDeepSpeech`
3. `pip install -r requirements.txt`
4. Get the data using the data download scripts in the download folder, LibriSpeech is a good example.
5. Download the language model (large file) run `./lm/get_lm.sh`

## Running
1. To Train, run `python run-train.py train_files <csvfile> valid_files <csvfile>` (see file for complete arguments list)
2. To Test, run `python run-test.py test_files <datacsvfile>`

<!-- ## iOS/Android -->
<!-- See iOS/Android folders -->

## Credit
1. Mozilla DeepSpeech
2. Baidu original DS1 & DS2 papers

<!-- ## Help -->
<!-- tbc -->

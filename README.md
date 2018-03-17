# Keras DeepSpeech
[![Build Status](https://travis-ci.org/robmsmt/KerasDeepSpeech.svg?branch=master)](https://travis-ci.org/robmsmt/KerasDeepSpeech)

<!-- ___ -->

Repository for experimenting with different CTC based model designs for ASR. Supports [live recording and testing](data/live-rec-test.py) of speech and quickly creates customised datasets using [own-voice dataset creation scripts](data/data-recorder.py)!

## OVERVIEW

<!-- ![Overview kDS](https://raw.githubusercontent.com/mlrobsmt/KerasDeepSpeech/master/preproc/overview.gif "Overview of kDS and batchgen") -->
<img src="https://raw.githubusercontent.com/robmsmt/KerasDeepSpeech/master/preproc/overview.gif" align="center" height="371" width="453">

<!-- ## Existing Architectures - model.py -->
<!-- 1. Arch 0 - DS1 (3FC+BLSTM+SOFTMAX) with dropout -->
<!-- 2. Arch 1 - DS1 (3FC+BLSTM+SOFTMAX) dropout -->
<!-- 3. Arch 2 - DS2 (1D conv+BGRU+FC+SOFTMAX) -->
<!-- 4. Arch 3 - own FC+ -->
<!-- 5. Arch 4 - Graves2006 (conv) -->
<!-- 6. Arch 2 - DS2 (conv) -->

<!-- ## QUICKSTART PRETRAINED MODELS -->


## SETUP
1. Recommended > use virtualenv installed with python2.7 (3.x untested and will not work with Core ML)
2. `git clone https://github.com/robmsmt/KerasDeepSpeech`
3. `pip install -r requirements.txt`
4. Get the data using the import/download scripts in the ![data](https://github.com/robmsmt/KerasDeepSpeech/tree/master/data) folder, LibriSpeech is a good example.
5. Download the language model (large file) run `./lm/get_lm.sh`

## RUN
1. To Train, simply run `python run-train.py` In order to specify training/validation files use `python run-train.py --train_files <csvfile> --valid_files <csvfile>` (see run-train for complete arguments list)
2. To Test, run `python run-test.py --test_files <datacsvfile>`

<!-- ## iOS/Android -->
<!-- See iOS/Android folders -->

## CREDIT 
1. Mozilla [DeepSpeech](https://github.com/mozilla/DeepSpeech)
2. Baidu [DS1](https://arxiv.org/abs/1412.5567) & [DS2](https://arxiv.org/abs/1512.02595) papers

<!-- ## Help -->
<!-- tbc -->

## Licence
The content of this project itself is licensed under the GNU General Public License.
Copyright © 2018

## Contributing ##
Have a question? Like the tool? Don't like it? Open an issue and let's talk about it! Pull requests are appreciated!


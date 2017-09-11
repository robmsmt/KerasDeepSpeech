# DOWNLOAD DATA


We've included import LibriSpeech and TIMIT in this directory to get started. All credits to the Mozilla team for the LibriSpeech download scripts.

For more, see Mozilla download scripts from [Mozilla DeepSpeech](https://github.com/mozilla/DeepSpeech/tree/master/bin)

## HOW TO RUN
1. cd to the root of the KerasDeepSpeech directory
2. run `./data/import_<dataset>.py data/`

## OTHER FREE DATASETS
[LibriVox](https://github.com/mozilla/DeepSpeech/blob/master/bin/import_librivox.py)

## LDC DATASETS
you need an account with academic institution otherwise these cost money

1. [TIMIT](https://catalog.ldc.upennedu/ldc93s1)

 NOTE to download LDC data to a server the following technique has been tested and works well:

 1. Download Cookie.txt export chrome extension [here](https://chrome.google.com/webstore/detail/cookietxt-export/lopabhfecdfhgogdbojmaicoicjekelh)
 2. Login to LDC, press cookie.txt and copy the content into cookie.txt on the server
 3. Use `wget -x --load-cookies cookies.txt https://catalog.ldc.upenn.edu/<LDCPATH>` to download the files.


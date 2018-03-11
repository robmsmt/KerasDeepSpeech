## Tensorflow Guide

### Please note: there is a problem with getting this to work on Android that I was not able to fix - this was due to the output dimension problem explained [here](https://stackoverflow.com/questions/45915517/output-dimension-issue-running-tensorflow-graph-android).


1. Download ALL of tensorflow (you need to compile TF for your mobile)
`git clone --recurse-submodules  https://github.com/tensorflow/tensorflow.git`

Important once downloaded change the branch to the one you are using for the model.
e.g. `git checkout r1.1`

2. Download Android SDK (comes with studio) and NDK:

- [SDK Link](https://developer.android.com/studio/index.html)
- [NDK Link](https://developer.android.com/ndk/index.html)


3. Install Bazel from here https://bazel.build/versions/master/docs/install.html


4. Edit workspace file at the root of the tensorflow direction you grabbed from git

Uncomment the lines about SDK/NDK

android_sdk_repository
android_ndk_repository

and fill in the correct paths to those libraries.


5. Build .so files

`bazel build -c opt //tensorflow/contrib/android:libtensorflow_inference.so \
   --crosstool_top=//external:android/crosstool \
   --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
   --cpu=armeabi-v7a`

the file is bazel-bin/tensorflow/contrib/android/libtensorflow_inference.so make a note of this

<!-- 6. Build java file -->

These need to be copied into the [TENSORFLOW KDS DROID APP](https://github.com/robmsmt/kds2Droid)


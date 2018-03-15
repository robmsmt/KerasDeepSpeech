## KDS on iOS

There are two methods to get kDS on iOS:

 1. Core ML - note this HAD an issue with RNNs it's been fixed so we will update this once it's been retested.

     1. Run `convert_keras_to_coreml.py` this will require setting arg to set the correct checkpoint path. This:
      1. Loads the pre-trained model into memory based on the checkpoint path
      2. Builds an identical model without the CTC Loss function or any non-compatible layers
      3. Transfers the weights using a numeric layer mapping (this might error if mapping is wrong)
      4. Saves and outputs the newly created model as a "trimmed model" to disk
      5. Calls coremltools convert and sets the metadata and saves file as `kDS.coreml`
      6. Copy the `kDS.coreml` into the Xcode app so that it can be used by the app

    An example of a swift4 Core ML app to load the model can be found at [github.com/robmsmt/kDS2iOS](https://github.com/mlrobsmt/kDS2iOS)

 2. TensorFlow graph - not covered in this guide - create_pb.py can export the TF graph for you.
 

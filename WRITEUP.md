# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

The process behind converting custom layers involves...we need to add extensions to both model optimizer and inference engine so that it can be handeled during inference.

Some of the potential reasons for handling custom layers are...to avoid errors reported by the inference engine.

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...

The difference between model accuracy pre- and post-conversion was...
pre and post conversion both gives the good and almost same results with slight differenve

# The size of the model pre- and post-conversion was...

- Model1: ssd_mobilenet_v2_coco:
  pre-conversion=202Mb and post-conversion=65Mb

- Model2: ssdlite_mobilenet_v2_coco:
  pre-conversion=60Mb and post-conversion=18Mb

# The inference time of the model pre- and post-conversion was...

- Model1: ssd_mobilenet_v2_coco:
  pre-conversion=203ms and post-conversion=78ms

- Model2: ssdlite_mobilenet_v2_coco:
  pre-conversion=156ms and post-conversion=53ms

## Assess Model Use Cases

Some of the potential use cases of the people counter app are...

1- in shopping marts. during rush hours people can be directed to other counters to avoid congession. the duration time also tells us how much time a customer spend in the store.

2- In hospital number of current admitted patients in a room can be counted and based on that we can determine that is there a bed avaliable for new patient.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

Lighting: Lighting conditions has to be good to correctly detect the persons. poor lighting conditions may result is not detecting the persons.
Model Accuracy: it depend's on different scenarios. in extreme cases Accuracy has to be high so that we can get better results.
Camera focal length: high resoultion camrea give accurate results. if the video quality of the camrea is low it will effect the accuracy.

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: ssd_mobilenet_v2_coco
  - http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
  - I converted the model to an Intermediate Representation with the following arguments...
  ```
  python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
  ```
  - The model is sufficient for the app because...the accuracy is good enough and inference time is suitable for the application
  - I tried to improve the model for the app by...i test the model on different threshold values and it give me best results at o.6. i run the application using following arguments.
  ```
  python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m ssd_mobilenet/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
  ```
- Model 2: ssdlite_mobilenet_v2_coco
  - http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
  - I converted the model to an Intermediate Representation with the following arguments...
  ```
  python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
  ```
  - The model is sufficient for the app because...the accuracy is lower than the first model but the inference is fast compared to first model
  - I tried to improve the model for the app by...i test the model on different threshold values and it give me best results at o.6. i run the application using following arguments.
  ```
  python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m ssdlite_mobilenet/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
  ```
## Conclusion
  - Model 1: ssd_mobilenet_v2_coco gave better results.

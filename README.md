# Spine-Transformers:Vertebra Detection and Localization in Arbitrary Field-of-View Spine CT with Transformers

This is the official implementation of the Spine-Transformers paper:

![alt text](https://github.com/gloriatao/Spine-Transformers/blob/main/images/Fig1_net_update.png)

In this paper, we address the problem of automatic detection and localization of vertebrae in arbitrary Field-Of-View (FOV) Spine CT. We propose a novel transformers-based 3D object detection method that views automatic detection of vertebrae in arbitrary FOV CT scans as an one-to-one set prediction problem. The main components of the new framework, called Spine-Transformers, are an one-to-one set based global loss that forces unique predictions and a light-weighted transformer architecture equipped with skip connections and learnable positional embeddings for encoder and decoder, respectively. It reasons about the relations of dierent levels of vertebrae and the global volume context to directly output all vertebrae in parallel. We additionally propose an inscribed sphere-based object detector to replace the regular box-based object detector for a better handling of volume orientation variation. Comprehensive experiments are conducted on two public datasets. The experimental results demonstrate the ecacy of the present approach.

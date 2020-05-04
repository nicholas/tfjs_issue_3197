/**
 * Sample React Native App
 * https://github.com/facebook/react-native
 *
 * @format
 * @flow strict-local
 */

import React from 'react';

import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';
import { asyncStorageIO, bundleResourceIO } from '@tensorflow/tfjs-react-native';

import { Asset } from 'expo-asset';
import * as FileSystem from 'expo-file-system';

import { base64ImageToTensor } from './image_utils';

const modelJson = require('./assets/converted_tfjs/model.json');
const modelWeights = require('./assets/converted_tfjs/group1-shard1of1.bin');
const imageURI = Asset.fromModule(require('./assets/images/tinyjpeg.jpg')).uri;

export class App extends React.Component {
  async componentDidMount() {
    await tf.setBackend('rn-webgl');
    await tf.ready();

    FileSystem.downloadAsync(imageURI, FileSystem.documentDirectory + 'image.jpg').then(async ({
        uri
    }) => {
        const base64 = await FileSystem.readAsStringAsync(uri, {
            "encoding": FileSystem.EncodingType.Base64
        })
        
        const model = await tf.loadGraphModel(bundleResourceIO(modelJson, modelWeights))
        const tensor = (await base64ImageToTensor(base64)).expandDims().toFloat()
        console.log("tensor", tensor)

        // Try 1
        // const result = await model.predict(tensor)
        // Exported tfjs model has one signature input, 'ToFloat'. Converted tfjs model has 2, 'key' & 'encoded_image_string_tensor'
        // => Error: Input tensor count mismatch,the graph model has 2 placeholders, while there are 1 input tensors.

        // Try 2
        // const result = await model.predict({ "ToFloat": tensor })
        // => Error: This execution contains the node 'Postprocessor/BatchMultiClassNonMaxSuppression/map/while/Exit_8', which has the dynamic op 'Exit'.Please use model.executeAsync() instead.Alternatively, to avoid the dynamic ops, specify the inputs[Postprocessor / BatchMultiClassNonMaxSuppression / map / TensorArrayStack_6 / TensorArrayGatherV3]

        // Try 3
        const result = await model.executeAsync({ "ToFloat": tensor })
        // Is _extremely_ slow depending on the source image
        // 2KB - 3+ minutes
        // 318KB - 10+ minutes
        // => TypeError: Unknown op 'NonMaxSuppressionV4'. File an issue at https://github.com/tensorflow/tfjs/issues so we can add it, or register a custom execution with tf.registerOp()

        const data = await result.data()
        console.log("data", data)
    }).catch(error => {
        console.error(error)
    });
  }

  render() {
    return null
  }
}

export default App;

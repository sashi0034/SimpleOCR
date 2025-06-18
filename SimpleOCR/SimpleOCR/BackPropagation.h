#pragma once
#include "Matrix.h"
#include "NeuralNetwork.h"

namespace ocr
{
    struct BackPropagationInput
    {
        NeuralNetworkInput neuralInput;

        int trueLabel;

        int batches;
    };

    struct BackPropagationOutput
    {
        /// @brief クロスエントロピー誤差
        float crossEntropyError;

        Matrix dw1; // [入力ノード数][中間ノード数]

        Array<float> db1; // [中間ノード数]

        Matrix dw2; // [中間ノード数][出力ノード数]

        Array<float> db2; // [出力ノード数]
    };

    BackPropagationOutput BackPropagation(const BackPropagationInput& input);
}

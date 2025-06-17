#pragma once
#include "Matrix.h"
#include "TY/Array.h"

namespace ocr
{
    struct NeuralNetworkInput
    {
        Array<float> x; // [入力ノード数]

        Matrix w1; // [入力ノード数][中間ノード数]

        Array<float> b1; // [中間ノード数]

        Matrix w2; // [中間ノード数][出力ノード数]

        Array<float> b2; // [出力ノード数]
    };

    struct NeuralNetworkOutput
    {
        Array<float> y1; // [中間ノード数]

        Array<float> y2; // [出力ノード数]

        const Array<float>& output() const
        {
            return y2;
        }

        int maxIndex() const;
    };

    NeuralNetworkOutput NeuralNetwork(const NeuralNetworkInput& input);
}

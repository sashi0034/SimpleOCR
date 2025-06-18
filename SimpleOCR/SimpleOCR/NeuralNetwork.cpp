#include "pch.h"
#include "NeuralNetwork.h"

#include "NP.h"

namespace
{
    Array<float> sigmoid(const Array<float>& a)
    {
        Array<float> y(a.size());
        for (int i = 0; i < a.size(); ++i)
        {
            // Sigmoid activation function
            y[i] = 1.0f / (1.0f + std::expf(-a[i]));
        }

        return y;
    }

    Array<float> softmax(const Array<float>& a)
    {
        float alpha = a[0];
        for (int i = 1; i < a.size(); ++i)
        {
            if (a[i] > alpha) alpha = a[i];
        }

        Array<float> y(a.size());
        float sum{};
        for (int i = 0; i < a.size(); ++i)
        {
            y[i] = std::expf(a[i] - alpha);
            sum += y[i];
        }

        for (int i = 0; i < a.size(); ++i)
        {
            y[i] = y[i] / sum;
        }

        return y;
    }
}

namespace ocr
{
    int NeuralNetworkOutput::maxIndex() const
    {
        int maxIndex = 0;
        const auto& output = y2;
        float maxValue = y2[0];
        for (int i = 1; i < output.size(); ++i)
        {
            if (output[i] > maxValue)
            {
                maxValue = output[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    NeuralNetworkOutput NeuralNetwork(const Array<float>& x, const NeuralNetworkParameters& params)
    {
        NeuralNetworkOutput output{};

        // x -> [w1 + b1] -> a1 -> sigmoid -> y1 -> [w2 + b2] -> a2 -> softmax -> y2 -> loss

        // ----------------------------------------------- 入力層 --> 中間層

        Array<float> a1 = params.b1;
        NP::GEMM(x, params.w1, a1); // a1 = x * w1 + b1

        // --> sigmoid 活性化関数層: 非線形性を加える
        output.y1 = sigmoid(a1);

        // ----------------------------------------------- 中間層 --> 出力層

        Array<float> a2 = params.b2;
        NP::GEMM(output.y1, params.w2, a2); // y2 = y1 * w2 + b2

        // --> softmax 活性化関数層: 出力を確率分布として解釈
        output.y2 = softmax(a2);

        return output;
    }
}

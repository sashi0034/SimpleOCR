#include "pch.h"
#include "BackPropagation.h"

#include "NP.h"
#include "TY/InlineComponent.h"

using namespace ocr;

namespace
{
    Array<float> oneHotEncoding(int index, int size)
    {
        assert(index >= 0 && index < size);

        Array<float> encoding(size, 0.0f);
        encoding[index] = 1.0f;
        return encoding;
    }

    float crossEntropyError(const Array<float>& predictedY, const Array<float>& trueY)
    {
        assert(predictedY.size() == trueY.size());

        float error = 0.0f;
        for (size_t i = 0; i < predictedY.size(); ++i)
        {
            error -= trueY[i] * std::logf(predictedY[i] + 1e-7f); // Add small value to avoid log(0)
        }

        return error;
    }

    BackPropagationOutput cpuBackPropagation(const BackPropagationInput& input)
    {
        BackPropagationOutput output{};
        const auto neuralOutput = NeuralNetwork(input.x, input.params);

        const Array<float>& x = input.x;

        const Array<float> trueY = oneHotEncoding(input.trueLabel, neuralOutput.output().size());

        output = {};

        // -----------------------------------------------

        // <-- softmax 逆伝搬
        const Array<float> da2 = NP::Divide(NP::Subtract(neuralOutput.y2, trueY), input.batches);

        output.dw2 = NP::OuterProduct(neuralOutput.y1, da2);

        output.db2 = da2;

        // -----------------------------------------------

        // <-- sigmoid 逆伝搬
        const Array<float> sigmoidGradient =
            NP::HadamardProduct(neuralOutput.y1,
                                NP::Subtract(Array<float>(neuralOutput.y1.size(), 1.0f), neuralOutput.y1));

        const Array<float> da1 = NP::HadamardProduct(
            NP::VecMat(da2, input.params.w2.transposed()).data(),
            sigmoidGradient
        );

        output.dw1 = NP::OuterProduct(x, da1);

        output.db1 = da1;

        // -----------------------------------------------

        output.crossEntropyError = crossEntropyError(neuralOutput.output(), trueY);
        return output;
    }

    // -----------------------------------------------

    struct GpuBackPropagation : IInlineComponent
    {
        bool initialized{};

        void EnsureInitialized()
        {
            if (initialized) return;
        }
    };

    InlineComponent<GpuBackPropagation> s_gpuBP{};

    BackPropagationOutput gpuBackPropagation(const BackPropagationInput& input)
    {
        s_gpuBP->EnsureInitialized();

        BackPropagationOutput output{};

        return output;
    }
}

namespace ocr
{
    BackPropagationOutput BackPropagation(const BackPropagationInput& input)
    {
        return cpuBackPropagation(input);
    }
}

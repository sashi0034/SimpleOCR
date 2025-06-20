#include "pch.h"
#include "BackPropagation.h"

#include "NP.h"
#include "TY/Gpgpu.h"
#include "TY/GpgpuBuffer.h"
#include "TY/InlineComponent.h"
#include "TY/Logger.h"
#include "TY/ScopedDeferStack.h"
#include "TY/Shader.h"

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

        ReadonlyGpgpuBufferView<float> y1{};
        ReadonlyGpgpuBufferView<float> da2{};
        WritableGpgpuBufferView<float> dw2{};
        WritableGpgpuBufferView<float> db2{};

        ReadonlyGpgpuBufferView<float> x{};
        ReadonlyGpgpuBufferView<float> w2{};
        WritableGpgpuBuffer1D<float> da1{};
        WritableGpgpuBufferView<float> dw1{};
        WritableGpgpuBufferView<float> db1{};

        ComputeShader csOuterProduct{ShaderParams::CS("asset/cs/outer_product.hlsl")};
        ComputeShader csSigmoidBackward{ShaderParams::CS("asset/cs/sigmoid_backward.hlsl")};

        Gpgpu outerProduct2{};
        Gpgpu sigmoidBackward{};
        Gpgpu outerProduct1{};

        void EnsureInitialized()
        {
            if (initialized) return;

            outerProduct2 = Gpgpu{
                GpgpuParams{}
                .setCS(csOuterProduct)
                .setReadonlyBuffer({y1, da2})
                .setWritableBuffer({dw2})
            };

            sigmoidBackward = Gpgpu{
                GpgpuParams{}
                .setCS(csSigmoidBackward)
                .setReadonlyBuffer({y1, w2, da2})
                .setWritableBuffer({da1, db1})
            };

            outerProduct1 = Gpgpu{
                GpgpuParams{}
                .setCS(csOuterProduct)
                .setReadonlyBuffer({x, da1.asReadonly()})
                .setWritableBuffer({dw1})
            };

            initialized = true;
        }
    };

    InlineComponent<GpuBackPropagation> s_gpuBP{};

    BackPropagationOutput gpuBackPropagation(const BackPropagationInput& input)
    {
        s_gpuBP->EnsureInitialized();

        BackPropagationOutput output{};
        const auto neuralOutput = NeuralNetwork(input.x, input.params);

        const Array<float>& x = input.x;

        const Array<float> trueY = oneHotEncoding(input.trueLabel, neuralOutput.output().size());

        // <-- softmax 逆伝搬 (最初の小規模な部分は CPU で計算)
        const Array<float> da2 = NP::Divide(NP::Subtract(neuralOutput.y2, trueY), input.batches);
        output.db2 = da2;

        output.dw2 = Matrix(neuralOutput.y1.size(), da2.size());
        output.db1 = Array<float>(neuralOutput.y1.size());
        output.dw1 = Matrix(x.size(), neuralOutput.y1.size());

        s_gpuBP->da1.resize(neuralOutput.y1.size());

        const auto scopedAssigns = ScopedDeferStack().push(
            s_gpuBP->y1.scopedReadonly(neuralOutput.y1),
            s_gpuBP->da2.scopedReadonly(da2),
            s_gpuBP->dw2.scopedWritable(output.dw2.data(), {output.dw2.rowsCols(), 1}),
            s_gpuBP->db2.scopedWritable(output.db2),
            s_gpuBP->x.scopedReadonly(x),
            s_gpuBP->w2.scopedReadonly(input.params.w2.data(), {input.params.w2.rowsCols(), 1}),
            s_gpuBP->dw1.scopedWritable(output.dw1.data(), {output.dw1.rowsCols(), 1}),
            s_gpuBP->db1.scopedWritable(output.db1)
        );

        Gpgpu::SequenceCompute({
            s_gpuBP->outerProduct2,
            s_gpuBP->sigmoidBackward,
            s_gpuBP->outerProduct1
        });

        output.crossEntropyError = crossEntropyError(neuralOutput.output(), trueY);

#if 1 // test
        const auto cpu = cpuBackPropagation(input);
        const bool ok_dw2 = output.dw2.data().sequenceAlmostEquals(cpu.dw2.data());
        const bool ok_db2 = output.db2.sequenceAlmostEquals(cpu.db2);
        const bool ok_dw1 = output.dw1.data().sequenceAlmostEquals(cpu.dw1.data());
        const bool ok_db1 = output.db1.sequenceAlmostEquals(cpu.db1);
        if (ok_dw2 && ok_db2 && ok_dw1 && ok_db1)
        {
            LogInfo.writeln("GpuBackPropagation: Test passed.");
        }
        else
        {
            LogError.writeln("GpuBackPropagation: Test failed.");
        }
#endif
        return output;
    }
}

namespace ocr
{
    BackPropagationOutput BackPropagation(const BackPropagationInput& input)
    {
        return gpuBackPropagation(input);
    }
}

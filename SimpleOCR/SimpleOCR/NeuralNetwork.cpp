#include "pch.h"
#include "NeuralNetwork.h"

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

    NeuralNetworkOutput cpuNeuralNetwork(const Array<float>& x, const NeuralNetworkParameters& params)
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

    // -----------------------------------------------

    struct GpuNeuralNetwork : IInlineComponent
    {
        bool initialized{};

        ReadonlyGpgpuBufferView<float> x{};
        ReadonlyGpgpuBufferView<float> w1{};
        ReadonlyGpgpuBufferView<float> b1{};
        ReadonlyGpgpuBufferView<float> w2{};
        ReadonlyGpgpuBufferView<float> b2{};

        WritableGpgpuBufferView<float> y1{};
        WritableGpgpuBufferView<float> y2{};

        ComputeShader csForwardLinear{ShaderParams::CS("asset/cs/forward_linear.hlsl")};
        ComputeShader csSigmoid{ShaderParams::CS("asset/cs/sigmoid.hlsl")};
        ComputeShader csSoftmax{ShaderParams::CS("asset/cs/softmax.hlsl")};

        Gpgpu forwardLinear1{};
        Gpgpu sigmoid{};
        Gpgpu forwardLinear2{};
        Gpgpu softmax{};

        void EnsureInitialized()
        {
            if (initialized) return;

            forwardLinear1 =
                GpgpuParams{}
                .setCS(csForwardLinear)
                .setReadonlyBuffer({x, w1, b1})
                .setWritableBuffer({y1});

            sigmoid =
                GpgpuParams{}
                .setCS(csSigmoid)
                .setWritableBuffer({y1});

            forwardLinear2 =
                GpgpuParams{}
                .setCS(csForwardLinear)
                .setReadonlyBuffer({y1.asReadonly(), w2, b2})
                .setWritableBuffer({y2});

            softmax =
                GpgpuParams{}
                .setCS(csSoftmax)
                .setWritableBuffer({y2});

            initialized = true;
        }
    };

    InlineComponent<GpuNeuralNetwork> s_gpuNN{};

    NeuralNetworkOutput gpuNeuralNetwork(const Array<float>& x, const NeuralNetworkParameters& params)
    {
        s_gpuNN->EnsureInitialized();

        NeuralNetworkOutput output{};
        output.y1.resize(params.w1.cols());
        output.y2.resize(params.w2.cols());

        constexpr int softmaxCapacity = 64;
        if (output.y2.size() >= softmaxCapacity)
        {
            LogError(std::format("GpuNeuralNetwork: y exceeds softmax capacity of {}.", softmaxCapacity));
        }

        const auto scopedAssigns = ScopedDeferStack().push(
            s_gpuNN->x.scopedReadonly(x),
            s_gpuNN->w1.scopedReadonly(params.w1.data()),
            s_gpuNN->b1.scopedReadonly(params.b1),
            s_gpuNN->w2.scopedReadonly(params.w2.data()),
            s_gpuNN->b2.scopedReadonly(params.b2),
            s_gpuNN->y1.scopedWritable(output.y1),
            s_gpuNN->y2.scopedWritable(output.y2)
        );
#if 0
        s_gpu->forwardLinear1.compute(); // a1 = x * w1 + b1
        s_gpu->sigmoid.compute();

        s_gpu->forwardLinear2.compute(); // a2 = y1 * w2 + b2
        s_gpu->softmax.compute();
#else // Optimize the above code by combining the compute calls
        Gpgpu::SequenceCompute({
            s_gpuNN->forwardLinear1,
            s_gpuNN->sigmoid,
            s_gpuNN->forwardLinear2,
            s_gpuNN->softmax
        });
#endif

#if 1 // test
        const auto cpu = cpuNeuralNetwork(x, params);
        if (output.y2.sequenceAlmostEquals(cpu.y2))
        {
            LogInfo.writeln("GpuNeuralNetwork: Test passed.");
        }
        else
        {
            LogError.writeln("GpuNeuralNetwork: Test failed.");
        }
#endif

        return output;
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
        return gpuNeuralNetwork(x, params);
    }
}

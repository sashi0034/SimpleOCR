#include "pch.h"
#include "NeuralNetwork.h"

#include "NP.h"
#include "TY/Gpgpu.h"
#include "TY/GpgpuBuffer.h"
#include "TY/InlineComponent.h"
#include "TY/Logger.h"
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

        ReadonlyGpgpuBuffer1D<float> x{};
        ReadonlyGpgpuBuffer2D<float> w1{};
        ReadonlyGpgpuBuffer1D<float> b1{};
        ReadonlyGpgpuBuffer2D<float> w2{};
        ReadonlyGpgpuBuffer1D<float> b2{};

        WritableGpgpuBuffer1D<float> y1{};
        WritableGpgpuBuffer1D<float> y2{};

        ComputeShader csForwardLinear{ShaderParams::CS("asset/cs/forward_linear.hlsl")};
        ComputeShader csSigmoid{ShaderParams::CS("asset/cs/sigmoid.hlsl")};
        ComputeShader csSoftmax{ShaderParams::CS("asset/cs/softmax.hlsl")};

        Gpgpu forwardLinear1{};
        Gpgpu sigmoid{};
        Gpgpu forwardLinear2{};
        Gpgpu softmax{};

        void EnsureInitialized(int x1Size, int y1ize, int y2Size)
        {
            if (initialized) return;

            x = ReadonlyGpgpuBuffer1D<float>(x1Size);
            w1 = ReadonlyGpgpuBuffer2D<float>(x1Size, y1ize);
            b1 = ReadonlyGpgpuBuffer1D<float>(y1ize);
            w2 = ReadonlyGpgpuBuffer2D<float>(y1ize, y2Size);
            b2 = ReadonlyGpgpuBuffer1D<float>(y2Size);

            y1 = WritableGpgpuBuffer1D<float>(y1ize);
            y2 = WritableGpgpuBuffer1D<float>(y2Size);

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

            constexpr int softmaxCapacity = 64;
            if (y2Size >= softmaxCapacity)
            {
                LogError(std::format("GpuNeuralNetwork: y exceeds softmax capacity of {}.", softmaxCapacity));
            }

            initialized = true;
        }
    };

    InlineComponent<GpuNeuralNetwork> s_gpu{};

    NeuralNetworkOutput gpuNeuralNetwork(const Array<float>& x, const NeuralNetworkParameters& params)
    {
        s_gpu->EnsureInitialized(x.size(), params.w1.cols(), params.w2.cols());

        s_gpu->x.data() = x;
        s_gpu->w1.data() = params.w1.data();
        s_gpu->b1.data() = params.b1;
        s_gpu->w2.data() = params.w2.data();
        s_gpu->b2.data() = params.b2;

        NeuralNetworkOutput output{};

        // x -> [w1 + b1] -> a1 -> sigmoid -> y1 -> [w2 + b2] -> a2 -> softmax -> y2 -> loss

#if 0
        s_gpu->forwardLinear1.compute(); // a1 = x * w1 + b1
        s_gpu->sigmoid.compute();

        s_gpu->forwardLinear2.compute(); // a2 = y1 * w2 + b2
        s_gpu->softmax.compute();
#else // Optimize the above
        Gpgpu::SequenceCompute({
            s_gpu->forwardLinear1,
            s_gpu->sigmoid,
            s_gpu->forwardLinear2,
            s_gpu->softmax
        });
#endif

        output.y1 = s_gpu->y1.data();
        output.y2 = s_gpu->y2.data();

#if 1 // test
        const auto cpu = cpuNeuralNetwork(x, params);
        if (output.y2.sequenceAlmostEquals(cpu.y2))
        {
            LogInfo.writeln("GpuNeuralNetwork: Output matches CPU implementation.");
        }
        else
        {
            LogError.writeln("GpuNeuralNetwork: Output does not match CPU implementation.");
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

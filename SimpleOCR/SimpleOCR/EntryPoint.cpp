#include "pch.h"
#include "EntryPoint.h"

#include "BackPropagation.h"
#include "DatasetImage.h"
#include "DatasetLoader.h"
#include "LivePPAddon.h"
#include "NeuralNetwork.h"
#include "TY/Gpgpu.h"
#include "TY/Image.h"
#include "TY/Logger.h"
#include "TY/Math.h"
#include "TY/Random.h"
#include "TY/System.h"
#include "TY/Texture.h"

using namespace TY;

using namespace ocr;

namespace
{
    constexpr int midNodeCount = 128;

    constexpr int labelCount = 10;

    constexpr int batchSize = 100;

    constexpr float learningRate = 0.01;

    constexpr int epochCount = 5;
}

struct EntryPointImpl
{
    ComputeShader m_computeShader{};
    WritableGpgpuBuffer1D<uint32_t> m_buffer{};
    ReadonlyGpgpuBuffer1D<uint32_t> m_readonlyData0{};
    ReadonlyGpgpuBuffer1D<uint32_t> m_readonlyData1{};
    Gpgpu m_gpgpu{};

    PixelShader m_texturePS{};
    VertexShader m_textureVS{};

    int m_trainImageIndex{};
    Texture m_previewTexture{};
    int m_predictedLabel{};

    DatasetImageList m_trainImages{};
    Array<uint8_t> m_trainLabels{};

    DatasetImageList m_testImage{};
    Array<uint8_t> m_testLabel{};

    NeuralNetworkParameters m_params{};

    EntryPointImpl()
    {
        m_computeShader = ComputeShader{ShaderParams::CS("asset/shader/simple_compute.hlsl")};

        m_buffer = WritableGpgpuBuffer1D<uint32_t>(100);
        m_readonlyData0 = ReadonlyGpgpuBuffer1D<uint32_t>(50);
        for (int i = 0; i < m_readonlyData0.data().size(); ++i)
        {
            m_readonlyData0.data()[i] = i * 10;
        }

        m_readonlyData1 = ReadonlyGpgpuBuffer1D<uint32_t>(100);
        for (int i = 0; i < m_readonlyData1.data().size(); ++i)
        {
            m_readonlyData1.data()[i] = -i;
        }

        m_gpgpu = Gpgpu{
            GpgpuParams{}
            .setCS(m_computeShader)
            .setWritableBuffer({m_buffer})
            .setReadonlyBuffer({m_readonlyData0, m_readonlyData1,})
        };

        m_gpgpu.compute();

        m_trainImages = LoadMnistImages("asset/dataset/train-images.idx3-ubyte");

        m_trainLabels = LoadMnistLabels("asset/dataset/train-labels.idx1-ubyte");

        m_testImage = LoadMnistImages("asset/dataset/t10k-images.idx3-ubyte");

        m_testLabel = LoadMnistLabels("asset/dataset/t10k-labels.idx1-ubyte");

        m_texturePS = PixelShader{ShaderParams::PS("asset/shader/default2d.hlsl")};
        m_textureVS = VertexShader{ShaderParams::VS("asset/shader/default2d.hlsl")};

        m_params = makeRandomNeuralInput(m_trainImages.images[0].size());

        m_trainImageIndex = 0;
        m_previewTexture = makePreviewTexture(m_trainImageIndex);
        m_predictedLabel = runNeuralNetwork(m_trainImageIndex);
    }

    void Update()
    {
        {
            m_previewTexture.as2D().scaled(4.0f).drawAt(Vec2{200, 200});
        }

        {
            ImGui::Begin("Train Images");

            if (ImGui::InputInt("Index", &m_trainImageIndex))
            {
                m_trainImageIndex =
                    Math::Clamp(m_trainImageIndex, 0, static_cast<int>(m_trainImages.images.size() - 1));
                m_previewTexture = makePreviewTexture(m_trainImageIndex);
                m_predictedLabel = runNeuralNetwork(m_trainImageIndex);
            }

            ImGui::Text("Actual Label: %d", m_trainLabels[m_trainImageIndex]);

            ImGui::Text("Predicted Label: %d", m_predictedLabel);

            if (ImGui::Button("Execute Machine Learning"))
            {
                machineLearning();
            }

            if (ImGui::Button("Compute Accuracy"))
            {
                computeAccuracy();
            }

            ImGui::End();
        }

        {
            ImGui::Begin("Compute Shader");

            const auto& data = m_buffer.data();
            ImGui::Text("Element Count: %d", data.size());

            ImGui::BeginGroup();
            for (size_t i = 0; i < data.size(); ++i)
            {
                ImGui::Text("[%d] = %u", i, data[i]);
                if (i % 4 == 3) ImGui::NewLine();
            }

            ImGui::EndGroup();

            if (ImGui::Button("Compute"))
            {
                m_gpgpu.compute();

                LogInfo.writeln("Computed!");
            }

            ImGui::End();
        }
    }

private:
    Texture makePreviewTexture(int index) const
    {
        return Texture{
            TextureParams()
            .setSource(m_trainImages.images[index].imageView(m_trainImages.property))
            .setPS(m_texturePS)
            .setVS(m_textureVS)
        };
    }

    NeuralNetworkParameters makeRandomNeuralInput(int inputRows) const
    {
        NeuralNetworkParameters neuralInput{};
        neuralInput = {};

        neuralInput.w1 = Matrix(inputRows, midNodeCount);
        neuralInput.w1.data() = neuralInput.w1.data().map([](uint8_t) { return Random::Float(-1.0f, 1.0f); });

        neuralInput.b1 = Array<float>(midNodeCount).map([](uint8_t) { return Random::Float(-1.0f, 1.0f); });

        neuralInput.w2 = Matrix(midNodeCount, labelCount);
        neuralInput.w2.data() = neuralInput.w2.data().map([](uint8_t) { return Random::Float(-1.0f, 1.0f); });

        neuralInput.b2 = Array<float>(labelCount).map([](uint8_t) { return Random::Float(-1.0f, 1.0f); });
        return neuralInput;
    }

    int runNeuralNetwork(int index)
    {
        const auto x = makeImageInput(m_trainImages.images[index]);
        const NeuralNetworkOutput neuralOutput = NeuralNetwork(x, m_params);
        return neuralOutput.maxIndex();
    }

    void acumulateGradients(NeuralNetworkParameters& params, const BackPropagationOutput& bp)
    {
        // Accumulate gradients for weights and biases
        for (int i = 0; i < params.w1.rows(); ++i)
        {
            for (int j = 0; j < params.w1.cols(); ++j)
            {
                params.w1[i][j] += bp.dw1[i][j];
            }
        }

        for (int i = 0; i < params.b1.size(); ++i)
        {
            params.b1[i] += bp.db1[i];
        }

        for (int i = 0; i < params.w2.rows(); ++i)
        {
            for (int j = 0; j < params.w2.cols(); ++j)
            {
                params.w2[i][j] += bp.dw2[i][j];
            }
        }

        for (int i = 0; i < params.b2.size(); ++i)
        {
            params.b2[i] += bp.db2[i];
        }
    }

    void backpropagationApply(NeuralNetworkParameters& params, const NeuralNetworkParameters& gradient)
    {
        // Update weights and biases using the gradients from backpropagation
        for (int i = 0; i < params.w1.rows(); ++i)
        {
            for (int j = 0; j < params.w1.cols(); ++j)
            {
                params.w1[i][j] -= learningRate * gradient.w1[i][j];
            }
        }

        for (int i = 0; i < params.b1.size(); ++i)
        {
            params.b1[i] -= learningRate * gradient.b1[i];
        }

        for (int i = 0; i < params.w2.rows(); ++i)
        {
            for (int j = 0; j < params.w2.cols(); ++j)
            {
                params.w2[i][j] -= learningRate * gradient.w2[i][j];
            }
        }

        for (int i = 0; i < params.b2.size(); ++i)
        {
            params.b2[i] -= learningRate * gradient.b2[i];
        }
    }

    void machineLearning()
    {
        Array<int> indices(m_trainImages.images.size());
        for (int i = 0; i < indices.size(); ++i)
        {
            indices[i] = i;
        }

        Random::Shuffle(indices);

        float previousAverageLoss{};
        constexpr float lossTermination = 0.01f;

        for (int epoch = 0; epoch < epochCount; ++epoch)
        {
            for (int i = 0; i < indices.size(); ++i)
            {
                indices[i] = i;
            }

            Random::Shuffle(indices);

            // -----------------------------------------------
            LogInfo.writeln(std::format("Epoch: {}/{}", epoch + 1, epochCount));

            float averageLoss = 0.0f;

            const int batchesPerEpoch = m_trainImages.images.size() / batchSize;
            for (int batch = 0; batch < batchesPerEpoch; ++batch)
            {
                const int baseIndex = batch * batchSize;

                NeuralNetworkParameters accGradient{};
                accGradient.w1 = Matrix(m_params.w1.rows(), m_params.w1.cols());
                accGradient.b1 = Array<float>(m_params.b1.size());
                accGradient.w2 = Matrix(m_params.w2.rows(), m_params.w2.cols());
                accGradient.b2 = Array<float>(m_params.b2.size());

                for (int i = 0; i < batchSize; ++i)
                {
                    const int imageIndex = indices[baseIndex + i];

                    auto x = makeImageInput(m_trainImages.images[imageIndex]);

                    const BackPropagationInput bpInput{
                        .x = std::move(x),
                        .params = m_params,
                        .trueLabel = m_trainLabels[imageIndex],
                        .batches = batchSize
                    };

                    const BackPropagationOutput bpOutput = BackPropagation(bpInput);

                    averageLoss += bpOutput.crossEntropyError;

                    acumulateGradients(accGradient, bpOutput);
                }

                backpropagationApply(m_params, accGradient);
            }

            averageLoss /= static_cast<float>(batchesPerEpoch * batchSize);

            LogInfo.writeln(std::format("Average Loss: {:.6f}", averageLoss));

            if (Abs(averageLoss - previousAverageLoss) < lossTermination)
            {
                LogInfo.writeln("Training terminated early due to low loss.");
                break;
            }

            previousAverageLoss = averageLoss;
        }
    }

    Array<float> makeImageInput(const DatasetImage& image) const
    {
        return image.map([](uint8_t pixel)
        {
            return static_cast<float>(pixel) / 255.0f;
        });
    }

    void computeAccuracy() const
    {
        int correctCount = 0;
        for (int i = 0; i < m_testImage.images.size(); ++i)
        {
            const auto x = makeImageInput(m_testImage.images[i]);

            const NeuralNetworkOutput neuralOutput = NeuralNetwork(x, m_params);
            if (neuralOutput.maxIndex() == m_testLabel[i])
            {
                correctCount++;
            }
        }

        const float accuracy = static_cast<float>(correctCount) / m_testImage.images.size();
        LogInfo.writeln(std::format("Training completed! Accuracy: {:.2f}", accuracy));
    }
};

void ocr::EntryPoint()
{
    EntryPointImpl impl{};

    while (System::Update())
    {
#ifdef _DEBUG
        Util::AdvanceLivePP();
#endif

        impl.Update();
    }
}

#include "pch.h"
#include "EntryPoint.h"

#include "BackPropagation.h"
#include "DatasetImage.h"
#include "DatasetLoader.h"
#include "ApplicationSettings.h"
#include "LivePPAddon.h"
#include "NeuralNetwork.h"
#include "TY/DynamicTexture.h"
#include "TY/Gpgpu.h"
#include "TY/Image.h"
#include "TY/Logger.h"
#include "TY/Math.h"
#include "TY/Mouse.h"
#include "TY/Random.h"
#include "TY/Scene.h"
#include "TY/Stopwatch.h"
#include "TY/System.h"
#include "TY/TextureDrawer.h"

using namespace TY;

using namespace ocr;

namespace
{
    constexpr int midNodeCount = 128;

    constexpr int labelCount = 10;

    constexpr int batchSize = 100;

    constexpr float learningRate = 0.01;

    constexpr int epochCount = 5;

    void drawline(Image& image, const Point& start, const Point& end, const ColorU8& color)
    {
        const int dx = end.x - start.x;
        const int dy = end.y - start.y;
        const int steps = Max(Abs(dx), Abs(dy));

        for (int i = 0; i <= steps; ++i)
        {
            const float t = static_cast<float>(i) / steps;
            const int x = static_cast<int>(start.x + dx * t);
            const int y = static_cast<int>(start.y + dy * t);
            if (image.inBounds(Point{x, y}) && color.r > image[Point{x, y}].r)
            {
                image[Point{x, y}] = color;
            }
        }
    }
}

struct EntryPointImpl
{
    PixelShader m_texturePS{};
    VertexShader m_textureVS{};

    int m_trainImageIndex{};
    TextureDrawer m_previewTexture{};
    int m_predictedLabel{};

    DatasetImageList m_trainImages{};
    Array<uint8_t> m_trainLabels{};

    DatasetImageList m_testImage{};
    Array<uint8_t> m_testLabel{};

    NeuralNetworkParameters m_params{};

    Image m_myImage{};
    DynamicTexture m_myTexture{};
    TextureDrawer m_myTextureDrawer{};
    int m_myImageLabel{};

    Array<std::string> m_epochMessages{};

    EntryPointImpl()
    {
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

        // -----------------------------------------------

        m_myImage = Image{m_trainImages.property.size, ColorF32{0.0f, 1.0f}.toColorU8()};
        m_myTexture = DynamicTexture{m_myImage};
        m_myTextureDrawer = TextureDrawer{
            TextureDrawerParams()
            .setSource(m_myTexture.getResource())
            .setPS(m_texturePS)
            .setVS(m_textureVS)
        };
    }

    void Update()
    {
        {
            m_previewTexture.as2D().scaled(4.0f).drawAt(Vec2{200, 200});
        }

        {
            constexpr float liveTextureScale = 12.0f;

            if (MouseL.pressed())
            {
                const auto topRight = Scene::Center() - m_myImage.size() * 0.5f * liveTextureScale;
                const auto canvasPos = (Mouse::PosF() - topRight) / liveTextureScale;
                const auto previousCanvasPos = (Mouse::PreviousPosF() - topRight) / liveTextureScale;
                if (canvasPos.inBounds(Size::Zero(), m_myImage.size() - Size::One()) ||
                    previousCanvasPos.inBounds(Size::Zero(), m_myImage.size() - Size::One()))
                {
                    // Draw a line from previous position to current position
                    for (int x = -2; x <= 2; x++)
                    {
                        for (int y = -2; y <= 2; y++)
                        {
                            const float c = 1.0f - Float2{x, y}.lengthSq() * 0.15f;
                            if (c < 0.0f) continue;

                            drawline(
                                m_myImage,
                                previousCanvasPos.asPoint() + Point{x, y},
                                canvasPos.asPoint() + Point{x, y},
                                ColorF32{c, 1.0f}.toColorU8());
                        }
                    }

                    m_myTexture.upload(m_myImage);

                    m_myImageLabel = runNeuralNetwork(makeImageInput(m_myImage));
                }
            }

            m_myTextureDrawer.as2D().scaled(liveTextureScale).drawAt(Scene::Center());

            {
                ImGui::Begin("My Image");

                ImGui::Separator();

                ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 255, 0, 255));
                ImGui::Text("Predicted Label: %d", m_myImageLabel);
                ImGui::PopStyleColor();

                ImGui::Separator();

                if (ImGui::Button("Clear"))
                {
                    m_myImage.fill(ColorF32{0.0f, 1.0f}.toColorU8());
                    m_myTexture.upload(m_myImage);
                }

                ImGui::End();
            }
        }

        {
            ImGui::Begin("Machine Learning");

            ImGui::Text("Current Train Image");

            if (ImGui::InputInt("Index", &m_trainImageIndex))
            {
                m_trainImageIndex =
                    Math::Clamp(m_trainImageIndex, 0, static_cast<int>(m_trainImages.images.size() - 1));
                m_previewTexture = makePreviewTexture(m_trainImageIndex);
                m_predictedLabel = runNeuralNetwork(m_trainImageIndex);
            }

            ImGui::Text("Actual Label: %d", m_trainLabels[m_trainImageIndex]);

            ImGui::Text("Predicted Label: %d", m_predictedLabel);

            ImGui::Separator();

            if (ImGui::Button("Execute Machine Learning"))
            {
                machineLearning();
            }

            static float s_accuracy{};

            if (ImGui::Button("Compute Accuracy"))
            {
                s_accuracy = computeAccuracy();
            }

            ImGui::Separator();
            ImGui::Text("Accuracy: %.2f%%", s_accuracy * 100.0f);
            ImGui::Separator();

            ImGui::End();
        }

        {
            ImGui::Begin("Epoch Messages");

            ImGui::Text("Epoch Messages:");

            for (const auto& message : m_epochMessages)
            {
                ImGui::TextUnformatted(message.c_str());
            }

            if (ImGui::Button("Clear Messages"))
            {
                m_epochMessages.clear();
            }

            ImGui::End();
        }

        {
            ImGui::Begin("Global Settings");

            ImGui::Checkbox("Use GPU", &g_applicationSettings.useGpu);

            ImGui::End();
        }
    }

private:
    TextureDrawer makePreviewTexture(int index) const
    {
        return TextureDrawer{
            TextureDrawerParams()
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

    int runNeuralNetwork(int index) const
    {
        return runNeuralNetwork(makeImageInput(m_trainImages.images[index]));
    }

    int runNeuralNetwork(const Array<float>& x) const
    {
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

            m_epochMessages.push_back(std::format("Epoch {}:\n- Average Loss = {:.6f}", epoch + 1, averageLoss));

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

    Array<float> makeImageInput(const Image& image) const
    {
        return image.data().map([](ColorU8 pixel)
        {
            return static_cast<float>(pixel.r) / 255.0f;
        });
    }

    float computeAccuracy() const
    {
        Stopwatch stopwatch{};

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
        LogInfo.writeln(std::format(
            "Training completed!\n- Accuracy: {:.2f}\n- Elapsed Time: {:.2f} seconds",
            accuracy,
            stopwatch.sF()));

        return accuracy;
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

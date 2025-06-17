#include "pch.h"
#include "EntryPoint.h"

#include "DatasetImage.h"
#include "DatasetLoader.h"
#include "LivePPAddon.h"
#include "TY/Gpgpu.h"
#include "TY/Image.h"
#include "TY/Logger.h"
#include "TY/Math.h"
#include "TY/System.h"
#include "TY/Texture.h"

using namespace TY;

using namespace ocr;

namespace
{
}

struct EntryPointImpl
{
    ComputeShader m_computeShader{};
    WritableGpgpuBuffer<uint32_t> m_buffer{};
    ReadonlyGpgpuBuffer<uint32_t> m_readonlyData0{};
    ReadonlyGpgpuBuffer<uint32_t> m_readonlyData1{};
    Gpgpu m_gpgpu{};

    PixelShader m_texturePS{};
    VertexShader m_textureVS{};

    Texture m_previewTexture{};

    DatasetImageList m_trainImages{};
    Array<uint8_t> m_trainLabels{};

    EntryPointImpl()
    {
        m_computeShader = ComputeShader{ShaderParams::CS("asset/shader/simple_compute.hlsl")};

        m_buffer = WritableGpgpuBuffer<uint32_t>(100);
        m_readonlyData0 = ReadonlyGpgpuBuffer<uint32_t>(50);
        for (int i = 0; i < m_readonlyData0.data().size(); ++i)
        {
            m_readonlyData0.data()[i] = i * 10;
        }

        m_readonlyData1 = ReadonlyGpgpuBuffer<uint32_t>(100);
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

        LoadMnistImages("asset/dataset/train-images.idx3-ubyte", m_trainImages);

        LoadMnistLabels("asset/dataset/train-labels.idx1-ubyte", m_trainLabels);

        m_texturePS = PixelShader{ShaderParams::PS("asset/shader/default2d.hlsl")};
        m_textureVS = VertexShader{ShaderParams::VS("asset/shader/default2d.hlsl")};

        m_previewTexture = makePreviewTexture(0);
    }

    void Update()
    {
        {
            m_previewTexture.as2D().scaled(4.0f).drawAt(Vec2{200, 200});
        }

        {
            ImGui::Begin("Train Images");

            static int s_trainImageIndex{};
            if (ImGui::InputInt("Index", &s_trainImageIndex))
            {
                s_trainImageIndex =
                    Math::Clamp(s_trainImageIndex, 0, static_cast<int>(m_trainImages.images.size() - 1));
                m_previewTexture = makePreviewTexture(s_trainImageIndex);
            }

            ImGui::Text("Actual Label: %d", m_trainLabels[s_trainImageIndex]);

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
    Texture makePreviewTexture(int index)
    {
        return Texture{
            TextureParams()
            .setSource(m_trainImages.images[index].imageView(m_trainImages.property))
            .setPS(m_texturePS)
            .setVS(m_textureVS)
        };
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

#include "pch.h"
#include "EntryPoint.h"

#include "DatasetImage.h"
#include "DatasetLoader.h"
#include "LivePPAddon.h"
#include "TY/Gpgpu.h"
#include "TY/Image.h"
#include "TY/Logger.h"
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

    Texture m_previewTexture{};

    DatasetImageList m_trainImages{};

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

        const auto texturePS = PixelShader{ShaderParams::PS("asset/shader/default2d.hlsl")};
        const auto textureVS = VertexShader{ShaderParams::VS("asset/shader/default2d.hlsl")};

        m_previewTexture = Texture{
            TextureParams()
            .setSource(m_trainImages.images[0].imageView(m_trainImages.property))
            .setPS(texturePS)
            .setVS(textureVS)
        };
    }

    void Update()
    {
        {
            m_previewTexture.drawAt(Vec2{200, 200});
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

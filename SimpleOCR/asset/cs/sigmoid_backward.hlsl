StructuredBuffer<float> g_y1 : register(t0);
StructuredBuffer<float> g_w2 : register(t1);
StructuredBuffer<float> g_da2 : register(t2);

RWStructuredBuffer<float> g_da1 : register(u0);
RWStructuredBuffer<float> g_db1 : register(u1);

cbuffer ReadonlyBufferSizes : register(b0)
{
    uint3 g_y1_size;
    uint3 g_w2_size;
    uint3 g_da2_size;
}

cbuffer WritableBufferSizes : register(b1)
{
    uint3 g_da1_size;
    uint3 g_db1_size;
}

[numthreads(64, 1, 1)]
void CS(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= g_da1_size.x) return;

    const float y1 = g_y1[DTid.x];
    const float sigmoidGradient = (1.0f - y1) * y1;

    float da1 = 0;
    for (int i = 0; i < g_da2_size.x; i++)
    {
        da1 += g_da2[i] * g_w2[DTid.x * g_da2_size.x + i];
    }

    da1 *= sigmoidGradient;

    g_da1[DTid.x] = da1;
    g_db1[DTid.x] = da1;
}

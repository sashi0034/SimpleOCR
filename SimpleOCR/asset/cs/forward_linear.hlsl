StructuredBuffer<float> g_x : register(t0);

StructuredBuffer<float> g_w : register(t1);

StructuredBuffer<float> g_b : register(t2);

RWStructuredBuffer<float> g_y : register(u0);

cbuffer ReadonlyBufferSizes : register(b0)
{
    uint3 g_x_size;
    uint3 g_w_size;
    uint3 g_b_size;
}

cbuffer WritableBufferSizes : register(b1)
{
    uint3 g_y_size;
}

[numthreads(64, 1, 1)]
void CS(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= g_y_size.x) return;

    float sum = 0.0f;
    for (uint i = 0; i < g_x_size.x; ++i)
    {
        sum += g_x[i] * g_w[i * g_y_size.x + DTid.x];
    }

    g_y[DTid.x] = sum + g_b[DTid.x];
}

RWStructuredBuffer<float> g_y : register(u0);

cbuffer WritableBufferSizes : register(b1)
{
    uint3 g_y_size;
}

[numthreads(64, 1, 1)]
void CS(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= g_y_size.x) return;

    g_y[DTid.x] = 1.0f / (1.0f + exp(-g_y[DTid.x])); // Sigmoid activation function
}

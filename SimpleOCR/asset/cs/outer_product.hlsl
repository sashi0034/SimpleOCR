StructuredBuffer<float> g_a : register(t0);
StructuredBuffer<float> g_b : register(t1);

RWStructuredBuffer<float> g_product : register(u0);

cbuffer ReadonlyBufferSizes : register(b0)
{
    uint3 g_a_size;
    uint3 g_b_size;
}

cbuffer WritableBufferSizes : register(b1)
{
    uint3 g_product_size;
}

[numthreads(8, 8, 1)]
void CS(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= g_product_size.x || DTid.y >= g_product_size.y) return;

    const int row = DTid.y * g_product_size.x;
    const int col = DTid.x;
    g_product[row + col] = g_a[DTid.y] * g_b[DTid.x]; // Outer product
}

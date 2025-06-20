RWStructuredBuffer<float> g_y : register(u0);

cbuffer WritableBufferSizes : register(b1)
{
    uint3 g_y_size;
}

groupshared float sharedData[64]; // Shared memory for input + exp values 
groupshared float sharedMax; // Maximum value
groupshared float sharedSum; // Sum of exponential

[numthreads(64, 1, 1)]
void CS(uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID)
{
    uint idx = DTid.x;
    uint localIdx = GTid.x;

    // Step 1: Copy input to shared memory and compute the max value
    sharedData[localIdx] = g_y[idx];

    GroupMemoryBarrierWithGroupSync(); // <-- barrier

    // Thread 0 calculates the maximum value
    if (localIdx == 0)
    {
        sharedMax = sharedData[0];
        for (uint i = 1; i < g_y_size.x; ++i)
        {
            sharedMax = max(sharedMax, sharedData[i]);
        }
    }

    GroupMemoryBarrierWithGroupSync(); // <-- barrier

    // Step 2: Compute exp(x - max)
    sharedData[localIdx] = exp(sharedData[localIdx] - sharedMax);

    GroupMemoryBarrierWithGroupSync(); // <-- barrier

    // Step 3: Thread 0 computes the sum of exponential
    if (localIdx == 0)
    {
        sharedSum = 0.0f;
        for (uint i = 0; i < g_y_size.x; ++i)
        {
            sharedSum += sharedData[i];
        }
    }

    GroupMemoryBarrierWithGroupSync(); // <-- barrier

    // Step 4: Write back the softmax result to g_y
    g_y[idx] = sharedData[localIdx] / sharedSum;
}

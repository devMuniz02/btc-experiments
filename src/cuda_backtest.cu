extern "C" __global__ void quant_stream_backtest_stub(
    const int* predictions,
    const float* probabilities,
    float* equity,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        equity[idx] = probabilities[idx] * static_cast<float>(predictions[idx]);
    }
}

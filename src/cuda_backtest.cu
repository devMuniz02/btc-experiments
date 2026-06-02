extern "C" __global__ void quant_stream_backtest_steps(
    const int* predictions,
    const int* targets,
    const int* trade_mask,
    float* step_return,
    int* win,
    int* signed_signal,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }

    int executed = trade_mask[idx];
    int pred = predictions[idx] == 1 ? 1 : 0;
    int signal = pred == 1 ? 1 : -1;
    int is_win = executed && (pred == targets[idx]);

    signed_signal[idx] = executed ? signal : 0;
    win[idx] = is_win ? 1 : 0;
    step_return[idx] = executed ? (is_win ? 1.0f : -1.0f) : 0.0f;
}

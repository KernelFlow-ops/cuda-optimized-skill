
1. 减少 __syncthreads() 次数：这是最直接的方式。很多时候开发者习惯性地插入同步，但如果warp内的数据访问模式天然不存在跨warp依赖，同步就是多余的。

2. Warp级同步替代Block级同步：一个warp内的32个线程天然是lockstep执行的（Volta之后有independent thread scheduling，但warp级原语仍然有效）。用 __syncwarp() 替代 __syncthreads() 可以把同步粒度从整个block缩小到单个warp，开销大幅降低。

3. Warp Shuffle（__shfl_sync 系列）：warp内线程之间交换数据完全不需要经过共享内存，没有bank conflict问题，延迟极低。适用于归约、前缀和、广播等模式。这直接消除了"写shared → sync → 读shared"的三步开销。

4. Cooperative Groups：CUDA 9引入的协作组机制，可以定义任意粒度的线程组并在该组内同步，比如只同步一个tile（比如8个线程），避免不必要的全block等待。


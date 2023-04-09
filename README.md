# Throughput comparison of cublasGemmEx

## Build
```
git clone https://github.com/enp1s0/gemmex-throughput
cd gemmex-throughput
git submodule update --init
make
```

## Run
```
# Usage: ./gemmex.test [min_log_N] [max_log_N] [mode list: FP32 FP16TC FP16TC_FP16DATA]

./gemmex.test 10 15 FP16TC FP16TC_FP16DATA FP32
```

## LICENSE

MIT

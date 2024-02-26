# How to run

1. Compile

```bash
nvcc K1_1_17_2_14.cu -o k1.out
```

2. Run

```bash
./k1.out "./testcases/q1_1000_1000.txt" "./testcases/k1_1000_1000_output.txt"
```

3. Profile

```bash
nvprof ./k1.out "./testcases/q1_1000_1000.txt" "./testcases/k1_1000_1000_output.txt"
```

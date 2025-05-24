main:
	g++ -O3 -march=native -mavx512f -Wall -fopenmp new_matmul.cpp -o matmul && objdump -d -M intel matmul > matmul_objdump.txt && ./matmul
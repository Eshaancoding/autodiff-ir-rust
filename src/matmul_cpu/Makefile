main:
	g++ -O3 -march=native -mavx512f -Wall -fopenmp matmul.cpp -o ./bin/matmul
	objdump -d -M intel ./bin/matmul > ./bin/matmul_objdump.txt 
	./bin/matmul

512:
	g++ -O3 -march=native -Wall -fopenmp 512_matmul.cpp -o ./bin/matmul
	objdump -d -M intel ./bin/matmul > ./bin/matmul_objdump.txt
	./bin/matmul
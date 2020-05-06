
output: main.o average.o
	g++ main.o average.o -o output

main.o: main.c
	g++ -c main.c

average.o: average.c average.h
	g++ -c average.c


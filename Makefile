exe:
	make clean
	make all

all: main

main: main.o
	g++ -ggdb -Wall -O3 -o main main.o

main.o: main.cpp
	g++ -ggdb -Wall -O3 -c main.cpp

dep:
	echo "Do nothing"

clean:
	rm -f main *.o

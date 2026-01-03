CXX = mpic++
CXXFLAGS = -std=c++17 -O3 -march=native -fopenmp
TARGET = poet_simulator
SRC = poet_simulator.cpp

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	mpirun -np 4 ./$(TARGET)

.PHONY: clean run
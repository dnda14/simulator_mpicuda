#ifndef BENCHMARK_DHT_HPP
#define BENCHMARK_DHT_HPP

#include <chrono>
#include <random>
#include <iostream>
#include <mpi.h>
#include "distributed_hash_table.hpp"

class DHTBenchmark {
private:
    DistributedHashTable& dht;
    int rank, size;
    
public:
    DHTBenchmark(DistributedHashTable& table, int rank, int size) 
        : dht(table), rank(rank), size(size) {}
    
    struct BenchmarkResult {
        double read_ops_per_sec;
        double write_ops_per_sec;
        double mixed_ops_per_sec;
        long long total_operations;
        double duration_ms;
    };
    
    BenchmarkResult runReadBenchmark(int operations_per_process) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, dht.getTotalCells() - 1);
        
        for (int i = 0; i < operations_per_process; ++i) {
            int cell_id = dis(gen);
            auto cell = dht.getCell(cell_id);
            // Prevenir optimizaciones del compilador
            asm volatile("" : "+r"(cell_id) : : "memory");
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        BenchmarkResult result;
        result.duration_ms = duration.count();
        result.total_operations = operations_per_process * size;
        result.read_ops_per_sec = (operations_per_process * size) / (duration.count() / 1000.0);
        
        return result;
    }
    
    BenchmarkResult runWriteBenchmark(int operations_per_process) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, dht.getTotalCells() - 1);
        
        for (int i = 0; i < operations_per_process; ++i) {
            int cell_id = dis(gen);
            GridCell new_cell(5); // 5 especies
            // Llenar con datos aleatorios
            for (auto& conc : new_cell.concentrations) {
                conc = static_cast<double>(rand()) / RAND_MAX;
            }
            dht.updateCell(cell_id, new_cell);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        BenchmarkResult result;
        result.duration_ms = duration.count();
        result.total_operations = operations_per_process * size;
        result.write_ops_per_sec = (operations_per_process * size) / (duration.count() / 1000.0);
        
        return result;
    }
    
    BenchmarkResult runMixedBenchmark(int operations_per_process, double read_ratio = 0.5) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, dht.getTotalCells() - 1);
        std::uniform_real_distribution<> prob_dis(0.0, 1.0);
        
        int reads = 0, writes = 0;
        
        for (int i = 0; i < operations_per_process; ++i) {
            int cell_id = dis(gen);
            
            if (prob_dis(gen) < read_ratio) {
                // Operación de lectura
                auto cell = dht.getCell(cell_id);
                reads++;
            } else {
                // Operación de escritura
                GridCell new_cell(5);
                for (auto& conc : new_cell.concentrations) {
                    conc = static_cast<double>(rand()) / RAND_MAX;
                }
                dht.updateCell(cell_id, new_cell);
                writes++;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        BenchmarkResult result;
        result.duration_ms = duration.count();
        result.total_operations = operations_per_process * size;
        result.mixed_ops_per_sec = (operations_per_process * size) / (duration.count() / 1000.0);
        
        if (rank == 0) {
            std::cout << "  Mixed operations - Reads: " << reads << ", Writes: " << writes << std::endl;
        }
        
        return result;
    }
    
    void printResults(const BenchmarkResult& result, const std::string& benchmark_name) {
        if (rank == 0) {
            std::cout << "=== " << benchmark_name << " ===" << std::endl;
            std::cout << "Duration: " << result.duration_ms << " ms" << std::endl;
            std::cout << "Total operations: " << result.total_operations << std::endl;
            
            if (result.read_ops_per_sec > 0) {
                std::cout << "Read operations/sec: " << result.read_ops_per_sec << std::endl;
            }
            if (result.write_ops_per_sec > 0) {
                std::cout << "Write operations/sec: " << result.write_ops_per_sec << std::endl;
            }
            if (result.mixed_ops_per_sec > 0) {
                std::cout << "Mixed operations/sec: " << result.mixed_ops_per_sec << std::endl;
            }
            std::cout << std::endl;
        }
    }
};

#endif // BENCHMARK_DHT_HPP
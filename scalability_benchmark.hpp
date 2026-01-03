#ifndef SCALABILITY_BENCHMARK_HPP
#define SCALABILITY_BENCHMARK_HPP

#include <vector>
#include <iostream>
#include <fstream>
#include <mpi.h>
#include "benchmark_dht.hpp"
#include "lock_free_hash_table.hpp"
#include "coarse_grained_hash_table.hpp"
#include "fine_grained_hash_table.hpp"

class ScalabilityBenchmark {
private:
    int rank, size;
    
public:
    ScalabilityBenchmark(int rank, int size) : rank(rank), size(size) {}
    
    struct ScalabilityResult {
        int processes;
        double lock_free_ops;
        double coarse_grained_ops;
        double fine_grained_ops;
        double speedup;
    };
    
    void runScalabilityStudy() {
        if (rank == 0) {
            std::cout << "\n🎯 RUNNING SCALABILITY STUDY" << std::endl;
            std::cout << "=============================" << std::endl;
        }
        
        std::vector<ScalabilityResult> results;
        const int BASE_OPERATIONS = 50000;
        
        SimulationParams params;
        params.grid_x = 500;
        params.grid_y = 1500;
        params.num_species = 5;
        int total_cells = params.grid_x * params.grid_y;
        
        // Ejecutar benchmark para el tamaño actual de MPI
        if (rank == 0) {
            std::cout << "Testing with " << size << " processes..." << std::endl;
        }
        
        // Lock-Free
        auto lock_free_table = std::make_unique<LockFreeHashTable>(params.num_species, total_cells, rank, size);
        DHTBenchmark lock_free_bench(*lock_free_table, rank, size);
        auto lock_free_result = lock_free_bench.runMixedBenchmark(BASE_OPERATIONS, 0.7);
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Coarse-Grained
        auto coarse_table = std::make_unique<CoarseGrainedHashTable>(params.num_species, total_cells, rank, size);
        DHTBenchmark coarse_bench(*coarse_table, rank, size);
        auto coarse_result = coarse_bench.runMixedBenchmark(BASE_OPERATIONS, 0.7);
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Fine-Grained
        auto fine_table = std::make_unique<FineGrainedHashTable>(params.num_species, total_cells, rank, size);
        DHTBenchmark fine_bench(*fine_table, rank, size);
        auto fine_result = fine_bench.runMixedBenchmark(BASE_OPERATIONS, 0.7);
        
        // Recolectar resultados en el proceso 0
        ScalabilityResult result;
        result.processes = size;
        result.lock_free_ops = lock_free_result.mixed_ops_per_sec;
        result.coarse_grained_ops = coarse_result.mixed_ops_per_sec;
        result.fine_grained_ops = fine_result.mixed_ops_per_sec;
        result.speedup = result.lock_free_ops / result.coarse_grained_ops; // CORREGIDO
        
        if (rank == 0) {
            results.push_back(result);
            printScalabilityResults(results);
            saveResultsToCSV(results);
        }
    }
    
    void printScalabilityResults(const std::vector<ScalabilityResult>& results) {
        std::cout << "\n📊 SCALABILITY RESULTS" << std::endl;
        std::cout << "====================" << std::endl;
        std::cout << "Procs | Lock-Free (ops/s) | Coarse (ops/s) | Fine (ops/s) | Speedup" << std::endl;
        std::cout << "------|-------------------|----------------|--------------|--------" << std::endl;
        
        for (const auto& result : results) {
            printf("%5d | %16.0f | %14.0f | %12.0f | %6.2fx\n",
                   result.processes,
                   result.lock_free_ops,
                   result.coarse_grained_ops,
                   result.fine_grained_ops,
                   result.speedup);
        }
    }
    
    void saveResultsToCSV(const std::vector<ScalabilityResult>& results) {
        if (rank == 0) {
            std::ofstream file("scalability_results.csv");
            file << "processes,lock_free_ops,coarse_grained_ops,fine_grained_ops,speedup\n";
            
            for (const auto& result : results) {
                file << result.processes << ","
                     << result.lock_free_ops << ","
                     << result.coarse_grained_ops << ","
                     << result.fine_grained_ops << ","
                     << result.speedup << "\n";
            }
            file.close();
            
            std::cout << "\n💾 Results saved to scalability_results.csv" << std::endl;
        }
    }
};

#endif // SCALABILITY_BENCHMARK_HPP
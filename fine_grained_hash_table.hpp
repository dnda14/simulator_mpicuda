#ifndef FINE_GRAINED_HASH_TABLE_HPP
#define FINE_GRAINED_HASH_TABLE_HPP

#include "distributed_hash_table.hpp"
#include <cstdint>

class FineGrainedHashTable : public DistributedHashTable {
public:
    FineGrainedHashTable(int total_entries, int rank, int size)
        : DistributedHashTable(total_entries, rank, size) {
        // También requiere época compartida
        MPI_Win_lock_all(MPI_MODE_NOCHECK, win);
    }

    ~FineGrainedHashTable() {
        MPI_Win_unlock_all(win);
    }

    void updateCell(int key, const GridCell& val) override {
        int target_rank = getOwnerRank(key);
        MPI_Aint base_offset = getLocalOffset(key);
        
        // Offset del campo "status" dentro del struct Bucket (usado como lock)
        MPI_Aint lock_offset = base_offset * sizeof(DHT_Bucket) + offsetof(DHT_Bucket, status); 
        
        // Usamos int porque status es int (4 bytes)
        int lock_val = 2;    // 2 = locked by writer
        int unlock_val = 0;  // 0 = empty/unlocked
        int occupied_val = 1; // 1 = occupied but unlocked
        int result_val = 0;
        bool locked = false;
        int max_attempts = 1000;
        int attempts = 0;

        // 1. ADQUIRIR LOCK (SPINLOCK REMOTO)
        while (!locked && attempts < max_attempts) {
            // Intentar cambiar de 0 o 1 a 2 (locked)
            // Primero intentamos con 0 (vacío)
            MPI_Compare_and_swap(&lock_val, &unlock_val, &result_val, 
                                 MPI_INT, target_rank, lock_offset, win);
            MPI_Win_flush(target_rank, win);
            
            if (result_val == 0) {
                locked = true;
            } else {
                // Intentamos con 1 (ocupado pero no locked)
                MPI_Compare_and_swap(&lock_val, &occupied_val, &result_val, 
                                     MPI_INT, target_rank, lock_offset, win);
                MPI_Win_flush(target_rank, win);
                if (result_val == 1) locked = true;
            }
            attempts++;
        }
        
        if (!locked) {
            // No pudimos adquirir el lock, abortamos esta operación
            return;
        }

        // 2. SECCIÓN CRÍTICA (Escribir datos)
        DHT_Bucket b;
        b.key = key; 
        b.value = val; 
        b.status = 1;  // Marcamos como ocupado (se sobrescribirá al final)
        b.checksum = 0;
        
        MPI_Put(&b, sizeof(DHT_Bucket), MPI_BYTE, 
                target_rank, base_offset * sizeof(DHT_Bucket),
                sizeof(DHT_Bucket), MPI_BYTE, win);
        MPI_Win_flush(target_rank, win);

        // 3. LIBERAR LOCK (poner status a 1 = ocupado pero libre)
        MPI_Accumulate(&occupied_val, 1, MPI_INT, target_rank, lock_offset, 
                       1, MPI_INT, MPI_REPLACE, win);
        MPI_Win_flush(target_rank, win);
    }
    
    // (getCell sería similar, adquiriendo el lock o usando Fetch_and_add como lector)

    GridCell getCell(int key) override {
        int target_rank = getOwnerRank(key);
        MPI_Aint base_offset = getLocalOffset(key);
        
        DHT_Bucket temp;
        
        // Lectura directa (el lock de escritura asegura que no leamos datos parciales)
        MPI_Get(&temp, sizeof(DHT_Bucket), MPI_BYTE,
                target_rank, base_offset * sizeof(DHT_Bucket),
                sizeof(DHT_Bucket), MPI_BYTE, win);
        MPI_Win_flush(target_rank, win);
        
        if (temp.status == 0 || temp.key != key) {
            return GridCell();
        }
        return temp.value;
    }

    std::string getStrategyName() const override {
        return "Fine-Grained (MPI_CAS)";
    }
};

#endif // FINE_GRAINED_HASH_TABLE_HPP
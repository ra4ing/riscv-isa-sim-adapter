// Copyright (c) 2024-2025 DiveFuzz Project
// SPDX-License-Identifier: Mulan PSL v2

#ifndef CHECKPOINT_H
#define CHECKPOINT_H

#include <cstdint>
#include <vector>
#include <map>
#include <memory>
#include <string>

// Forward declarations
class processor_t;
class sim_t;

namespace spike_engine {

//==============================================================================
// Load Reservation State (for LR/SC atomic operations)
//==============================================================================

/**
 * Load reservation state for atomic operations
 */
struct ReservationState {
    bool valid;                 // Whether a reservation is active
    uint64_t address;           // Reserved physical address
    static constexpr uint64_t INVALID_ADDR = static_cast<uint64_t>(-1);

    ReservationState() : valid(false), address(INVALID_ADDR) {}

    void clear() {
        valid = false;
        address = INVALID_ADDR;
    }
};

//==============================================================================
// Checkpoint Structure - Complete processor state snapshot
//==============================================================================

/**
 * Checkpoint: Complete processor state snapshot
 *
 * Represents a complete snapshot of processor state at a specific point.
 * Designed for efficient rollback during instruction validation.
 *
 * Design Philosophy:
 *   - Complete state: Saves all architectural state for precise rollback
 *   - Fast restore: Optimized for common validation patterns
 *   - Memory safety: Includes memory region backup for store rollback
 *
 * Memory Consistency:
 *   After restore, all architectural state is exactly as at checkpoint time.
 *   Instruction memory may contain newer instructions, but this is safe because
 *   PC points to the checkpoint position.
 */
struct Checkpoint {
    //==========================================================================
    // Core Architectural State
    //==========================================================================

    // General purpose registers (x0-x31)
    std::vector<uint64_t> xpr;

    // Floating point registers (f0-f31)
    // freg_t has two 64-bit fields: v[0] and v[1]
    // Both must be saved/restored for correct NaN-boxing behavior
    std::vector<uint64_t> fpr;      // FPR[i].v[0] - main 64-bit value
    std::vector<uint64_t> fpr_v1;   // FPR[i].v[1] - extended/internal state

    // Program counter
    uint64_t pc;

    //==========================================================================
    // Execution Position State
    //==========================================================================

    // Index into instruction sequence
    // Marks how many instructions have been executed at checkpoint time
    size_t instr_index;

    // Next available memory address for instruction placement
    uint64_t next_instruction_addr;

    //==========================================================================
    // Memory State
    //==========================================================================

    // Memory region backup for checkpoint/restore
    // Essential for correct rollback when instructions modify memory
    // (e.g., AMO instructions, store instructions)
    std::vector<uint8_t> mem_region_backup;

    //==========================================================================
    // Privilege and Mode State
    //==========================================================================

    // Privilege level (0=U, 1=S, 3=M)
    uint64_t prv;

    // Virtualization mode (for H extension)
    bool v;

    // Debug mode flag
    bool debug_mode;

    //==========================================================================
    // CSR State
    //==========================================================================

    // Complete CSR state captured as address->value map
    // Ensures ALL CSRs are properly restored
    std::map<uint64_t, uint64_t> csr_values;

    //==========================================================================
    // Extended Architectural State
    //==========================================================================

    // Load reservation state (for LR/SC atomic operations)
    ReservationState reservation;

    // Privilege transition tracking
    uint64_t prev_prv;
    bool prev_v;

    // Vector register file (if V extension enabled)
    std::vector<uint8_t> vector_regfile;

    // Timer CSRs serialization flag
    bool serialized;

    // Expected Landing Pad state (Zicfilp extension)
    // 0 = NO_LP_EXPECTED, 1 = LP_EXPECTED
    uint8_t elp;

    // Single step state (debug mode)
    // 0 = STEP_NONE, 1 = STEP_STEPPING, 2 = STEP_STEPPED
    uint8_t single_step;

    // Critical error flag
    bool critical_error;

    //==========================================================================
    // Metadata
    //==========================================================================

    // Whether this checkpoint is valid
    bool valid;

    // Checkpoint creation timestamp (optional, for debugging)
    uint64_t timestamp;

    // Checkpoint ID (for multi-checkpoint scenarios)
    uint32_t id;

    //==========================================================================
    // Constructor
    //==========================================================================

    Checkpoint();

    //==========================================================================
    // Utility Methods
    //==========================================================================

    /**
     * Clear all checkpoint data
     */
    void clear();

    /**
     * Check if checkpoint contains valid data
     */
    bool is_valid() const { return valid; }

    /**
     * Get approximate memory usage in bytes
     */
    size_t memory_usage() const;
};

//==============================================================================
// CheckpointManager - Manages checkpoint save/restore operations
//==============================================================================

/**
 * CheckpointManager: Handles checkpoint save/restore operations
 *
 * Separates checkpoint logic from the main execution engine.
 * Provides clean interface for state capture and restoration.
 *
 * All state saving/restoring is handled internally - no external dependencies.
 */
class CheckpointManager {
public:
    /**
     * Constructor
     * @param proc Processor pointer
     * @param sim Simulator pointer
     */
    CheckpointManager(processor_t* proc, sim_t* sim);

    /**
     * Set memory region for backup
     * @param start Start address of memory region
     * @param size Size of memory region in bytes
     */
    void set_memory_region(uint64_t start, size_t size);

    //==========================================================================
    // Single Checkpoint Operations
    //==========================================================================

    /**
     * Save current state to checkpoint
     * @param checkpoint Checkpoint to save to
     * @param instr_index Current instruction index
     * @param next_instr_addr Next instruction address
     */
    void save(Checkpoint& checkpoint, size_t instr_index, uint64_t next_instr_addr);

    /**
     * Restore state from checkpoint
     * @param checkpoint Checkpoint to restore from
     * @param[out] instr_index Restored instruction index
     * @param[out] next_instr_addr Restored next instruction address
     */
    void restore(const Checkpoint& checkpoint, size_t& instr_index, uint64_t& next_instr_addr);

    //==========================================================================
    // Multi-Checkpoint Support (for advanced use cases)
    //==========================================================================

    /**
     * Create a new checkpoint with unique ID
     * @param instr_index Current instruction index
     * @param next_instr_addr Next instruction address
     * @return Checkpoint ID
     */
    uint32_t create_checkpoint(size_t instr_index, uint64_t next_instr_addr);

    /**
     * Get checkpoint by ID
     * @param id Checkpoint ID
     * @return Pointer to checkpoint, or nullptr if not found
     */
    Checkpoint* get_checkpoint(uint32_t id);
    const Checkpoint* get_checkpoint(uint32_t id) const;

    /**
     * Restore from checkpoint by ID
     * @param id Checkpoint ID
     * @param[out] instr_index Restored instruction index
     * @param[out] next_instr_addr Restored next instruction address
     * @return true if successful, false if checkpoint not found
     */
    bool restore_checkpoint(uint32_t id, size_t& instr_index, uint64_t& next_instr_addr);

    /**
     * Delete checkpoint by ID
     * @param id Checkpoint ID
     * @return true if deleted, false if not found
     */
    bool delete_checkpoint(uint32_t id);

    /**
     * Clear all checkpoints
     */
    void clear_all_checkpoints();

    /**
     * Get number of active checkpoints
     */
    size_t checkpoint_count() const { return checkpoints_.size(); }

    /**
     * Get list of checkpoint IDs
     */
    std::vector<uint32_t> get_checkpoint_ids() const;

private:
    processor_t* proc_;
    sim_t* sim_;

    // Memory region configuration
    uint64_t mem_region_start_;
    size_t mem_region_size_;

    // Multi-checkpoint storage
    std::map<uint32_t, Checkpoint> checkpoints_;
    uint32_t next_checkpoint_id_;

    //==========================================================================
    // Internal Helper Methods
    //==========================================================================

    /**
     * Save registers to checkpoint
     */
    void save_registers(Checkpoint& checkpoint);

    /**
     * Restore registers from checkpoint
     */
    void restore_registers(const Checkpoint& checkpoint);

    /**
     * Save CSRs to checkpoint
     */
    void save_csrs(Checkpoint& checkpoint);

    /**
     * Restore CSRs from checkpoint
     */
    void restore_csrs(const Checkpoint& checkpoint);

    /**
     * Save memory region to checkpoint
     */
    void save_memory(Checkpoint& checkpoint);

    /**
     * Restore memory region from checkpoint
     */
    void restore_memory(const Checkpoint& checkpoint);

    /**
     * Save privilege/mode state
     */
    void save_privilege_state(Checkpoint& checkpoint);

    /**
     * Restore privilege/mode state
     */
    void restore_privilege_state(const Checkpoint& checkpoint);

    /**
     * Save extended architectural state (reservation, vector regs, etc.)
     */
    void save_extended_state(Checkpoint& checkpoint);

    /**
     * Restore extended architectural state
     */
    void restore_extended_state(const Checkpoint& checkpoint);
};

} // namespace spike_engine

#endif // CHECKPOINT_H

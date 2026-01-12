// Copyright (c) 2024-2025 DiveFuzz Project
// SPDX-License-Identifier: Mulan PSL v2

#ifndef SPIKE_ENGINE_H
#define SPIKE_ENGINE_H

#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <map>

#include "state_query.h"
#include "checkpoint.h"

// Forward declarations
class sim_t;
class processor_t;
class cfg_t;

namespace spike_engine {

/**
 * SpikeEngine: Efficient Spike execution engine with checkpointing
 *
 * A lightweight wrapper around the Spike RISC-V ISA simulator, designed for
 * instruction-by-instruction execution with support for mixed-length instructions
 * (16-bit compressed and 32-bit standard) and efficient checkpoint/restore.
 *
 * Key Features:
 *   - Unified execution model: One method handles all cases (single, jump, loop)
 *   - Mixed-length instruction support (RV*C compressed + standard)
 *   - Dynamic address allocation (no pre-allocated slots)
 *   - Lightweight checkpointing (registers + index only)
 *   - Python layer handles XOR validation (simpler, more flexible)
 *
 * Typical Workflow:
 *   1. Initialize with pre-compiled ELF template (containing nop placeholder region)
 *   2. Engine executes template initialization code until reaching main()
 *   3. For each instruction to test:
 *      a. set_checkpoint() - save current processor state
 *      b. Read source register values via StateQuery
 *      c. execute_sequence() - write and execute instruction(s)
 *      d. Read destination register values via StateQuery
 *      e. Python computes XOR and checks uniqueness
 *      f. If collision: restore_checkpoint() and retry
 *         If unique: proceed to next instruction
 *
 * Memory Management:
 *   Instructions are written sequentially starting from main symbol address.
 *   2-byte compressed instructions are detected automatically and written with
 *   correct size. PC tracking ensures proper alignment for mixed-length sequences.
 *
 * Thread Safety:
 *   Not thread-safe. Each thread should use its own SpikeEngine instance.
 *
 * Architecture:
 *   - SpikeEngine: Execution control (initialize, checkpoint, execute)
 *   - StateQuery: All processor state queries (registers, CSRs, memory, etc.)
 *   - CheckpointManager: Checkpoint save/restore operations
 *
 * Example (C++):
 *   SpikeEngine engine("template.elf", "rv64gc", 1000);
 *   engine.initialize();
 *   auto* sq = engine.get_state_query();
 *
 *   engine.set_checkpoint();
 *   uint64_t src_val = sq->get_xpr(2);  // Read x2 before
 *   engine.execute_sequence({0x003100b3}, {4});  // add x1, x2, x3
 *   uint64_t dst_val = sq->get_xpr(1);  // Read x1 after
 *
 * Example (Python):
 *   engine = SpikeEngine("template.elf", "rv64gc", 1000)
 *   engine.initialize()
 *   sq = engine.get_state_query()
 *
 *   engine.set_checkpoint()
 *   src_val = sq.get_xpr(2)
 *   engine.execute_sequence([0x003100b3], [4])
 *   dst_val = sq.get_xpr(1)
 */
class SpikeEngine {
public:
    /**
     * Constructor
     * @param elf_path Path to pre-compiled ELF file
     * @param isa ISA string (e.g., "rv64gc")
     * @param num_instrs Number of instructions to generate (nops in ELF)
     * @param verbose Enable verbose output (default: false)
     */
    SpikeEngine(const std::string& elf_path,
                const std::string& isa,
                size_t num_instrs,
                bool verbose = false);

    /**
     * Destructor
     */
    ~SpikeEngine();

    /**
     * Initialize Spike and run until main function entry
     * This executes template initialization code
     * @return true on success, false on error
     */
    bool initialize();

    /**
     * Create a checkpoint of current processor state
     * Saves PC, registers, and modified memory
     */
    void set_checkpoint();

    /**
     * Restore processor state from last checkpoint
     */
    void restore_checkpoint();

    /**
     * Execute a sequence of instructions
     *
     * Unified execution method that handles all cases:
     * - Single instruction: execute_sequence([code], [size])
     * - Forward jump: execute_sequence([jump, middle...], [sizes...])
     * - Backward loop: execute_sequence([init, body..., decr, branch], [sizes...])
     *
     * Execution logic:
     * 1. Write all instructions to memory starting at next_instruction_addr_
     * 2. Calculate target_pc = next_instruction_addr_ + sum(sizes)
     * 3. Execute until PC == target_pc
     * 4. Update next_instruction_addr_ to target_pc
     *
     * All cases are handled uniformly:
     * - Sequential: PC advances to target_pc after each step
     * - Trap: Spike handles internally, mret returns to next instruction
     * - Forward jump: PC jumps directly to target
     * - Backward loop: PC loops until branch falls through
     *
     * @param machine_codes List of machine codes to execute
     * @param sizes List of instruction sizes (2 or 4 bytes each)
     * @param max_steps Maximum execution steps (safety limit, default: 10000)
     * @return Number of instructions in the sequence
     *
     * @throws std::runtime_error if:
     *   - Engine not initialized
     *   - Out of instruction region space
     *   - PC mismatch (internal consistency error)
     *   - Memory write failure
     *   - max_steps exceeded before reaching target_pc
     */
    size_t execute_sequence(
        const std::vector<uint32_t>& machine_codes,
        const std::vector<size_t>& sizes,
        size_t max_steps = 10000);

    //==========================================================================
    // Engine Configuration Getters
    //==========================================================================

    /**
     * Get mem_region start address
     * @return Start address of mem_region (for testing memory operations)
     */
    uint64_t get_mem_region_start() const { return mem_region_start_; }

    /**
     * Get mem_region size
     * @return Size of mem_region in bytes
     */
    size_t get_mem_region_size() const { return mem_region_size_; }

    /**
     * Get current instruction index
     * @return Index of next instruction to replace
     */
    size_t get_current_index() const { return current_instr_index_; }

    /**
     * Get total number of instructions
     */
    size_t get_num_instrs() const { return num_instrs_; }

    /**
     * Get last error message
     */
    std::string get_last_error() const { return last_error_; }

    /**
     * Check if the last executed instruction triggered a trap/exception.
     * This is useful for logging - instructions that cause traps are handled
     * by the exception handler (which skips them), but they are still "accepted"
     * from the fuzzer's perspective.
     *
     * @return true if the last instruction triggered a trap, false otherwise
     */
    bool was_last_execution_trapped() const { return last_execution_trapped_; }

    /**
     * Get the number of trap handler steps executed in the last execution.
     * Returns 0 if no trap occurred.
     *
     * @return Number of steps executed in trap handler
     */
    size_t get_last_trap_handler_steps() const { return last_trap_handler_steps_; }

    //==========================================================================
    // Modular Component Accessors
    //==========================================================================

    /**
     * Get the StateQuery interface for all processor state queries
     *
     * StateQuery provides comprehensive CPU state query interfaces:
     * - Basic registers: get_xpr(), get_fpr(), get_pc(), get_all_xpr(), get_all_fpr()
     * - CSRs: get_csr(), get_all_csrs()
     * - Memory: read_mem()
     * - Commit log: get_commit_log()
     * - Privilege: get_privilege_state()
     * - Trap info: get_last_trap_info()
     * - Vector: get_vector_state(), is_vector_enabled()
     * - Reservation: get_reservation_state(), has_reservation(), clear_reservation()
     * - Debug: get_debug_state()
     *
     * @return Pointer to StateQuery object (valid for lifetime of SpikeEngine)
     */
    StateQuery* get_state_query() { return state_query_.get(); }
    const StateQuery* get_state_query() const { return state_query_.get(); }

    /**
     * Get the CheckpointManager for advanced checkpoint operations
     * @return Pointer to CheckpointManager (valid for lifetime of SpikeEngine)
     */
    CheckpointManager* get_checkpoint_manager() { return checkpoint_manager_.get(); }
    const CheckpointManager* get_checkpoint_manager() const { return checkpoint_manager_.get(); }

    /**
     * Detect instruction size from machine code (static utility)
     *
     * Determines whether an instruction is 16-bit compressed or 32-bit standard
     * based on RISC-V encoding rules specified in the ISA manual.
     *
     * RISC-V Instruction Length Encoding (bits[1:0]):
     *   - bits[1:0] != 0b11  →  16-bit compressed instruction (RV*C)
     *   - bits[1:0] == 0b11  →  32-bit or longer instruction
     *     - If bits[4:2] != 0b111  →  32-bit standard instruction
     *     - If bits[4:2] == 0b111  →  48/64-bit instruction (not supported)
     *
     * @param machine_code 32-bit instruction encoding (may contain zero-padded
     *                     16-bit compressed instruction in lower half)
     *
     * @return Instruction size in bytes: 2 for compressed, 4 for standard
     *
     * @throws std::runtime_error if instruction length > 32 bits (unsupported)
     *
     * @note This is a static method and can be called without an instance:
     *       size_t sz = SpikeEngine::get_instruction_size(0x00008522);
     */
    static size_t get_instruction_size(uint32_t machine_code);

private:
    // Configuration
    std::string elf_path_;
    std::string isa_;
    size_t num_instrs_;
    bool verbose_;

    // Spike simulator
    std::unique_ptr<cfg_t> cfg_;
    std::unique_ptr<sim_t> sim_;
    processor_t* proc_;  // Main processor (core 0)

    // Instruction region
    uint64_t instruction_region_start_;
    uint64_t instruction_region_end_;
    uint64_t next_instruction_addr_;  // Next available address for instruction

    // Memory region for checkpoint/restore (e.g., .mem_region section)
    uint64_t mem_region_start_;
    size_t mem_region_size_;

    // Execution state
    size_t current_instr_index_;

    Checkpoint checkpoint_;  // Single checkpoint for simple use case
    bool initialized_;

    // Error handling
    std::string last_error_;

    // Trap detection (for logging)
    bool last_execution_trapped_;      // True if last instruction triggered a trap
    size_t last_trap_handler_steps_;   // Steps executed in trap handler (0 if no trap)

    // State query interface (modular state access)
    std::unique_ptr<StateQuery> state_query_;

    // Checkpoint manager (modular checkpoint operations)
    std::unique_ptr<CheckpointManager> checkpoint_manager_;

    // Internal helper methods

    /**
     * Find instruction region using ELF symbols
     * Reads 'main' symbol from ELF to locate instruction region
     */
    bool find_nop_region();

    /**
     * Read symbol address from ELF file
     * Uses objdump to extract symbol address
     * @param symbol_name Symbol name to look up (e.g., "main")
     * @return Symbol address, or 0 if not found
     */
    uint64_t read_symbol_address(const std::string& symbol_name);

    /**
     * Write machine code to memory address
     * @param addr Physical address
     * @param code Machine code
     * @param size Instruction size (2 or 4 bytes)
     */
    bool write_memory(uint64_t addr, uint32_t code, size_t size);

    /**
     * Read memory from physical address
     * @param addr Physical address
     * @return 32-bit value
     */
    uint32_t read_memory(uint64_t addr);

    /**
     * Single step execution (used during initialization)
     * Executes one instruction on processor
     */
    bool step_processor();

};

} // namespace spike_engine

#endif // SPIKE_ENGINE_H

// Copyright (c) 2024-2025 DiveFuzz Project
// SPDX-License-Identifier: Mulan PSL v2

#ifndef STATE_QUERY_H
#define STATE_QUERY_H

#include <cstdint>
#include <vector>
#include <map>
#include <string>
#include <optional>

#include "checkpoint.h"  // For ReservationState

// Forward declarations
class processor_t;
class mmu_t;
class sim_t;

namespace spike_engine {

//==============================================================================
// Commit Log Structures - Track per-instruction side effects
//==============================================================================

/**
 * Memory access record for commit log
 */
struct MemoryAccess {
    uint64_t addr;      // Memory address
    uint64_t value;     // Value read/written
    uint8_t size;       // Access size in bytes (1, 2, 4, 8)

    MemoryAccess() : addr(0), value(0), size(0) {}
    MemoryAccess(uint64_t a, uint64_t v, uint8_t s) : addr(a), value(v), size(s) {}
};

/**
 * Register write record for commit log
 */
struct RegisterWrite {
    uint16_t reg_num;   // Register number (0-31 for XPR, 32-63 for FPR, >=4096 for CSR)
    uint64_t value;     // New value written

    RegisterWrite() : reg_num(0), value(0) {}
    RegisterWrite(uint16_t r, uint64_t v) : reg_num(r), value(v) {}
};

/**
 * Complete commit log for a single instruction
 * Captures all side effects of instruction execution
 */
struct CommitLog {
    std::vector<RegisterWrite> reg_writes;  // Registers modified
    std::vector<MemoryAccess> mem_reads;    // Memory reads performed
    std::vector<MemoryAccess> mem_writes;   // Memory writes performed
    uint64_t inst_priv;                      // Privilege level during execution
    int inst_xlen;                           // XLEN during execution
    int inst_flen;                           // FLEN during execution

    CommitLog() : inst_priv(3), inst_xlen(64), inst_flen(64) {}

    void clear() {
        reg_writes.clear();
        mem_reads.clear();
        mem_writes.clear();
    }

    bool empty() const {
        return reg_writes.empty() && mem_reads.empty() && mem_writes.empty();
    }
};

//==============================================================================
// Privilege State - Track privilege mode transitions
//==============================================================================

/**
 * Privilege and virtualization mode state
 */
struct PrivilegeState {
    uint64_t prv;       // Current privilege level (0=U, 1=S, 3=M)
    uint64_t prev_prv;  // Previous privilege level
    bool prv_changed;   // Privilege changed on last instruction
    bool v;             // Virtualization mode (H extension)
    bool prev_v;        // Previous virtualization mode
    bool v_changed;     // Virtualization mode changed on last instruction
    bool debug_mode;    // Currently in debug mode

    PrivilegeState()
        : prv(3), prev_prv(3), prv_changed(false)
        , v(false), prev_v(false), v_changed(false)
        , debug_mode(false) {}

    // Helper to get privilege name
    const char* prv_name() const {
        switch (prv) {
            case 0: return "U";
            case 1: return "S";
            case 3: return "M";
            default: return "?";
        }
    }
};

//==============================================================================
// Debug and Execution State
//==============================================================================

/**
 * Debug and execution control state
 */
struct DebugState {
    bool debug_mode;        // Currently in debug mode
    uint8_t single_step;    // Single step state (0=NONE, 1=STEPPING, 2=STEPPED)
    bool critical_error;    // Critical error occurred
    uint8_t elp;            // Expected Landing Pad (0=NO_LP_EXPECTED, 1=LP_EXPECTED)
    bool serialized;        // Timer CSRs in well-defined state

    DebugState()
        : debug_mode(false), single_step(0), critical_error(false)
        , elp(0), serialized(true) {}

    // Helper to get single_step name
    const char* single_step_name() const {
        switch (single_step) {
            case 0: return "NONE";
            case 1: return "STEPPING";
            case 2: return "STEPPED";
            default: return "?";
        }
    }

    // Helper to get elp name
    const char* elp_name() const {
        switch (elp) {
            case 0: return "NO_LP_EXPECTED";
            case 1: return "LP_EXPECTED";
            default: return "?";
        }
    }
};

//==============================================================================
// Trap/Exception Information
//==============================================================================

/**
 * Detailed trap/exception information
 */
struct TrapInfo {
    bool occurred;      // Whether a trap occurred
    uint64_t cause;     // Trap cause code
    uint64_t tval;      // Trap value (bad address/instruction)
    uint64_t tval2;     // Second trap value (guest physical address)
    uint64_t tinst;     // Trapped instruction encoding
    bool has_gva;       // Has guest virtual address
    std::string name;   // Human-readable trap name

    TrapInfo()
        : occurred(false), cause(0), tval(0), tval2(0), tinst(0)
        , has_gva(false), name("") {}

    void clear() {
        occurred = false;
        cause = 0;
        tval = 0;
        tval2 = 0;
        tinst = 0;
        has_gva = false;
        name.clear();
    }
};

//==============================================================================
// Vector Unit State
//==============================================================================

/**
 * Vector extension internal state (beyond CSRs)
 */
struct VectorState {
    uint64_t vl;        // Vector length
    uint64_t vtype;     // Vector type register
    uint64_t vstart;    // Vector start index
    uint64_t vxsat;     // Vector saturation flag
    uint64_t vxrm;      // Vector rounding mode
    uint64_t vlenb;     // Vector register length in bytes

    // Derived/internal state
    uint64_t vlmax;     // Maximum vector length for current config
    uint64_t vsew;      // Selected element width (8, 16, 32, 64)
    float vflmul;       // Fractional LMUL value
    uint64_t vma;       // Mask agnostic flag
    uint64_t vta;       // Tail agnostic flag
    bool vill;          // Illegal configuration flag
    uint64_t VLEN;      // Hardware VLEN
    uint64_t ELEN;      // Hardware ELEN

    // Vector register file (optional, for complete state)
    std::vector<uint8_t> vreg_file;  // Complete vector register file

    VectorState()
        : vl(0), vtype(0), vstart(0), vxsat(0), vxrm(0), vlenb(0)
        , vlmax(0), vsew(0), vflmul(0), vma(0), vta(0), vill(false)
        , VLEN(0), ELEN(0) {}
};

//==============================================================================
// StateQuery Class - Main interface for state queries
//==============================================================================

/**
 * StateQuery provides rich CPU state query interfaces
 *
 * This class encapsulates all state query functionality, separating it
 * from the core execution engine. It provides:
 * - Basic register queries (XPR, FPR, PC)
 * - CSR queries
 * - Memory queries
 * - Commit log access (per-instruction side effects)
 * - Privilege state tracking
 * - Trap/exception information
 * - Vector unit state
 * - Load reservation state
 */
class StateQuery {
public:
    /**
     * Constructor
     * @param proc Processor pointer
     * @param sim Simulator pointer (for debug MMU access)
     */
    StateQuery(processor_t* proc, sim_t* sim);

    //==========================================================================
    // Basic Register Queries
    //==========================================================================

    /**
     * Get value of a general-purpose register
     * @param reg_index Register index (0-31)
     * @return Register value
     */
    uint64_t get_xpr(int reg_index) const;

    /**
     * Get all general-purpose register values
     * @return Vector of 32 register values (x0-x31)
     */
    std::vector<uint64_t> get_all_xpr() const;

    /**
     * Get value of a floating-point register
     * @param reg_index Register index (0-31)
     * @return Register value (as uint64_t, lower 64 bits)
     */
    uint64_t get_fpr(int reg_index) const;

    /**
     * Get all floating-point register values
     * @return Vector of 32 register values (f0-f31)
     */
    std::vector<uint64_t> get_all_fpr() const;

    /**
     * Get program counter value
     * @return PC value
     */
    uint64_t get_pc() const;

    //==========================================================================
    // CSR Queries
    //==========================================================================

    /**
     * Get a CSR value by address
     * @param csr_addr CSR address (e.g., 0x300 for mstatus)
     * @return CSR value, or 0 if not found/accessible
     */
    uint64_t get_csr(uint64_t csr_addr) const;

    /**
     * Get all accessible CSR values
     * @return Map of CSR address -> value
     */
    std::map<uint64_t, uint64_t> get_all_csrs() const;

    //==========================================================================
    // Memory Queries
    //==========================================================================

    /**
     * Read memory at specified address
     * @param addr Memory address to read from
     * @param size Number of bytes to read
     * @return Vector of bytes read from memory
     */
    std::vector<uint8_t> read_mem(uint64_t addr, size_t size) const;

    //==========================================================================
    // Commit Log Queries
    //==========================================================================

    /**
     * Get the commit log from the last executed instruction
     * @return CommitLog structure with all side effects
     */
    CommitLog get_commit_log() const;

    /**
     * Clear the commit log (call before executing next instruction)
     */
    void clear_commit_log();

    //==========================================================================
    // Privilege State Queries
    //==========================================================================

    /**
     * Get current privilege state including transition flags
     * @return PrivilegeState structure
     */
    PrivilegeState get_privilege_state() const;

    /**
     * Check if privilege level changed on last instruction
     * @return true if privilege changed
     */
    bool did_privilege_change() const;

    //==========================================================================
    // Debug State Queries
    //==========================================================================

    /**
     * Get debug and execution control state
     * @return DebugState structure
     */
    DebugState get_debug_state() const;

    /**
     * Check if in debug mode
     * @return true if in debug mode
     */
    bool is_debug_mode() const;

    /**
     * Check if single stepping
     * @return true if single stepping
     */
    bool is_single_stepping() const;

    /**
     * Check if critical error occurred
     * @return true if critical error
     */
    bool has_critical_error() const;

    /**
     * Get Expected Landing Pad state (Zicfilp extension)
     * @return 0 = NO_LP_EXPECTED, 1 = LP_EXPECTED
     */
    uint8_t get_elp() const;

    //==========================================================================
    // Trap/Exception Queries
    //==========================================================================

    /**
     * Get trap information from last execution
     * @return TrapInfo structure (check .occurred field)
     */
    TrapInfo get_last_trap_info() const;

    /**
     * Set trap info (called by execute when trap occurs)
     */
    void set_trap_info(const TrapInfo& info);

    /**
     * Clear trap info
     */
    void clear_trap_info();

    //==========================================================================
    // Vector Unit Queries
    //==========================================================================

    /**
     * Get vector unit state
     * @param include_regfile If true, include complete vector register file
     * @return VectorState structure
     */
    VectorState get_vector_state(bool include_regfile = false) const;

    /**
     * Check if vector extension is enabled
     * @return true if V extension is available
     */
    bool is_vector_enabled() const;

    //==========================================================================
    // Load Reservation Queries
    //==========================================================================

    /**
     * Get load reservation state
     * @return ReservationState structure (defined in checkpoint.h)
     */
    ReservationState get_reservation_state() const;

    /**
     * Check if a load reservation is currently active
     * @return true if reservation is valid
     */
    bool has_reservation() const;

    /**
     * Clear the load reservation (yield)
     */
    void clear_reservation();

private:
    processor_t* proc_;
    sim_t* sim_;
    mutable TrapInfo last_trap_;  // Cached trap info from last execution
};

} // namespace spike_engine

#endif // STATE_QUERY_H

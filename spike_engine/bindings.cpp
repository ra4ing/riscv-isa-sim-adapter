// Copyright (c) 2024-2025 DiveFuzz Project
// SPDX-License-Identifier: Mulan PSL v2

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "spike_engine.h"
#include "state_query.h"
#include "checkpoint.h"

namespace py = pybind11;
using namespace spike_engine;

PYBIND11_MODULE(spike_engine, m) {
    m.doc() = "Efficient Spike execution engine with checkpointing for DiveFuzz";

    // Floating-point register index offset
    // Register index convention:
    // - 0-31: Integer registers (x0-x31)
    // - 32-63: Floating-point registers (f0-f31, use FPR_OFFSET + reg_num)
    m.attr("FPR_OFFSET") = 32;

    //==========================================================================
    // State Query Structures (NEW)
    //==========================================================================

    // MemoryAccess structure
    py::class_<MemoryAccess>(m, "MemoryAccess",
        "Memory access record for commit log")
        .def(py::init<>())
        .def_readwrite("addr", &MemoryAccess::addr, "Memory address")
        .def_readwrite("value", &MemoryAccess::value, "Value read/written")
        .def_readwrite("size", &MemoryAccess::size, "Access size in bytes (1, 2, 4, 8)");

    // RegisterWrite structure
    py::class_<RegisterWrite>(m, "RegisterWrite",
        "Register write record for commit log")
        .def(py::init<>())
        .def_readwrite("reg_num", &RegisterWrite::reg_num,
            "Register number (0-31 for XPR, 32-63 for FPR, >=4096 for CSR)")
        .def_readwrite("value", &RegisterWrite::value, "New value written");

    // CommitLog structure
    py::class_<CommitLog>(m, "CommitLog",
        "Complete commit log for a single instruction")
        .def(py::init<>())
        .def_readwrite("reg_writes", &CommitLog::reg_writes, "Registers modified")
        .def_readwrite("mem_reads", &CommitLog::mem_reads, "Memory reads performed")
        .def_readwrite("mem_writes", &CommitLog::mem_writes, "Memory writes performed")
        .def_readwrite("inst_priv", &CommitLog::inst_priv, "Privilege level during execution")
        .def_readwrite("inst_xlen", &CommitLog::inst_xlen, "XLEN during execution")
        .def_readwrite("inst_flen", &CommitLog::inst_flen, "FLEN during execution")
        .def("clear", &CommitLog::clear, "Clear the commit log")
        .def("empty", &CommitLog::empty, "Check if commit log is empty");

    // PrivilegeState structure
    py::class_<PrivilegeState>(m, "PrivilegeState",
        "Privilege and virtualization mode state")
        .def(py::init<>())
        .def_readwrite("prv", &PrivilegeState::prv, "Current privilege level (0=U, 1=S, 3=M)")
        .def_readwrite("prev_prv", &PrivilegeState::prev_prv, "Previous privilege level")
        .def_readwrite("prv_changed", &PrivilegeState::prv_changed,
            "Privilege changed on last instruction")
        .def_readwrite("v", &PrivilegeState::v, "Virtualization mode (H extension)")
        .def_readwrite("prev_v", &PrivilegeState::prev_v, "Previous virtualization mode")
        .def_readwrite("v_changed", &PrivilegeState::v_changed,
            "Virtualization mode changed on last instruction")
        .def_readwrite("debug_mode", &PrivilegeState::debug_mode, "Currently in debug mode");

    // TrapInfo structure
    py::class_<TrapInfo>(m, "TrapInfo",
        "Detailed trap/exception information")
        .def(py::init<>())
        .def_readwrite("occurred", &TrapInfo::occurred, "Whether a trap occurred")
        .def_readwrite("cause", &TrapInfo::cause, "Trap cause code")
        .def_readwrite("tval", &TrapInfo::tval, "Trap value (bad address/instruction)")
        .def_readwrite("tval2", &TrapInfo::tval2, "Second trap value (guest physical address)")
        .def_readwrite("tinst", &TrapInfo::tinst, "Trapped instruction encoding")
        .def_readwrite("has_gva", &TrapInfo::has_gva, "Has guest virtual address")
        .def_readwrite("name", &TrapInfo::name, "Human-readable trap name")
        .def("clear", &TrapInfo::clear, "Clear the trap info");

    // VectorState structure
    py::class_<VectorState>(m, "VectorState",
        "Vector extension internal state")
        .def(py::init<>())
        .def_readwrite("vl", &VectorState::vl, "Vector length")
        .def_readwrite("vtype", &VectorState::vtype, "Vector type register")
        .def_readwrite("vstart", &VectorState::vstart, "Vector start index")
        .def_readwrite("vxsat", &VectorState::vxsat, "Vector saturation flag")
        .def_readwrite("vxrm", &VectorState::vxrm, "Vector rounding mode")
        .def_readwrite("vlenb", &VectorState::vlenb, "Vector register length in bytes")
        .def_readwrite("vlmax", &VectorState::vlmax, "Maximum vector length for current config")
        .def_readwrite("vsew", &VectorState::vsew, "Selected element width (8, 16, 32, 64)")
        .def_readwrite("vflmul", &VectorState::vflmul, "Fractional LMUL value")
        .def_readwrite("vma", &VectorState::vma, "Mask agnostic flag")
        .def_readwrite("vta", &VectorState::vta, "Tail agnostic flag")
        .def_readwrite("vill", &VectorState::vill, "Illegal configuration flag")
        .def_readwrite("VLEN", &VectorState::VLEN, "Hardware VLEN")
        .def_readwrite("ELEN", &VectorState::ELEN, "Hardware ELEN")
        .def_readwrite("vreg_file", &VectorState::vreg_file, "Complete vector register file");

    // ReservationState structure
    py::class_<ReservationState>(m, "ReservationState",
        "Load reservation state for atomic operations (LR/SC)")
        .def(py::init<>())
        .def_readwrite("valid", &ReservationState::valid, "Whether a reservation is active")
        .def_readwrite("address", &ReservationState::address, "Reserved physical address")
        .def("clear", &ReservationState::clear, "Clear the reservation")
        .def_readonly_static("INVALID_ADDR", &ReservationState::INVALID_ADDR,
            "Constant representing invalid/no reservation");

    // DebugState structure
    py::class_<DebugState>(m, "DebugState",
        "Debug and execution control state")
        .def(py::init<>())
        .def_readwrite("debug_mode", &DebugState::debug_mode, "Currently in debug mode")
        .def_readwrite("single_step", &DebugState::single_step,
            "Single step state (0=NONE, 1=STEPPING, 2=STEPPED)")
        .def_readwrite("critical_error", &DebugState::critical_error, "Critical error occurred")
        .def_readwrite("elp", &DebugState::elp,
            "Expected Landing Pad (0=NO_LP_EXPECTED, 1=LP_EXPECTED)")
        .def_readwrite("serialized", &DebugState::serialized, "Timer CSRs in well-defined state")
        .def("single_step_name", &DebugState::single_step_name,
            "Get human-readable single_step name")
        .def("elp_name", &DebugState::elp_name,
            "Get human-readable elp name");

    //==========================================================================
    // Checkpoint class (updated)
    //==========================================================================

    // Checkpoint class
    py::class_<Checkpoint>(m, "Checkpoint")
        .def(py::init<>())
        .def_readwrite("xpr", &Checkpoint::xpr, "General-purpose registers (x0-x31)")
        .def_readwrite("fpr", &Checkpoint::fpr, "Floating-point registers (f0-f31)")
        .def_readwrite("pc", &Checkpoint::pc, "Program counter")
        .def_readwrite("instr_index", &Checkpoint::instr_index, "Current instruction index")
        .def_readwrite("prv", &Checkpoint::prv, "Privilege level")
        .def_readwrite("v", &Checkpoint::v, "Virtualization mode")
        .def_readwrite("debug_mode", &Checkpoint::debug_mode, "Debug mode flag");

    //==========================================================================
    // StateQuery class - Advanced state query interface
    //==========================================================================

    py::class_<StateQuery>(m, "StateQuery",
        R"pbdoc(
        StateQuery provides comprehensive CPU state query interfaces.

        This class encapsulates all state query functionality:
        - Basic register queries (XPR, FPR, PC)
        - CSR queries
        - Memory queries
        - Commit log access (per-instruction side effects)
        - Privilege state tracking
        - Trap/exception information
        - Vector unit state
        - Load reservation state

        Typically accessed via SpikeEngine.get_state_query(), not instantiated directly.
        )pbdoc")

        // Basic Register Queries
        .def("get_xpr", &StateQuery::get_xpr,
             py::arg("reg_index"),
             "Get general-purpose register value (x0-x31)")

        .def("get_all_xpr", &StateQuery::get_all_xpr,
             "Get all general-purpose register values (x0-x31)")

        .def("get_fpr", &StateQuery::get_fpr,
             py::arg("reg_index"),
             "Get floating-point register value (f0-f31)")

        .def("get_all_fpr", &StateQuery::get_all_fpr,
             "Get all floating-point register values (f0-f31)")

        .def("get_pc", &StateQuery::get_pc,
             "Get program counter value")

        // CSR Queries
        .def("get_csr", &StateQuery::get_csr,
             py::arg("csr_addr"),
             "Get CSR value by address (e.g., 0x300 for mstatus)")

        .def("get_all_csrs", &StateQuery::get_all_csrs,
             "Get all accessible CSR values as dict {addr: value}")

        // Memory Queries
        .def("read_mem", &StateQuery::read_mem,
             py::arg("addr"),
             py::arg("size"),
             "Read memory at specified address, returns list of bytes")

        // Commit Log Queries
        .def("get_commit_log", &StateQuery::get_commit_log,
             "Get commit log from last executed instruction")

        .def("clear_commit_log", &StateQuery::clear_commit_log,
             "Clear the commit log (call before executing next instruction)")

        // Privilege State Queries
        .def("get_privilege_state", &StateQuery::get_privilege_state,
             "Get current privilege state including transition flags")

        .def("did_privilege_change", &StateQuery::did_privilege_change,
             "Check if privilege level changed on last instruction")

        // Debug State Queries
        .def("get_debug_state", &StateQuery::get_debug_state,
             "Get debug and execution control state")

        .def("is_debug_mode", &StateQuery::is_debug_mode,
             "Check if in debug mode")

        .def("is_single_stepping", &StateQuery::is_single_stepping,
             "Check if single stepping")

        .def("has_critical_error", &StateQuery::has_critical_error,
             "Check if critical error occurred")

        .def("get_elp", &StateQuery::get_elp,
             "Get Expected Landing Pad state (Zicfilp extension)")

        // Trap/Exception Queries
        .def("get_last_trap_info", &StateQuery::get_last_trap_info,
             "Get trap information from last execution")

        .def("set_trap_info", &StateQuery::set_trap_info,
             py::arg("info"),
             "Set trap info (called by execute when trap occurs)")

        .def("clear_trap_info", &StateQuery::clear_trap_info,
             "Clear trap info")

        // Vector Unit Queries
        .def("get_vector_state", &StateQuery::get_vector_state,
             py::arg("include_regfile") = false,
             "Get vector unit state")

        .def("is_vector_enabled", &StateQuery::is_vector_enabled,
             "Check if vector extension is enabled")

        // Load Reservation Queries
        .def("get_reservation_state", &StateQuery::get_reservation_state,
             "Get load reservation state for atomic operations")

        .def("has_reservation", &StateQuery::has_reservation,
             "Check if a load reservation is currently active")

        .def("clear_reservation", &StateQuery::clear_reservation,
             "Clear the load reservation (yield)");

    // SpikeEngine class
    py::class_<SpikeEngine>(m, "SpikeEngine")
        .def(py::init<const std::string&, const std::string&, size_t, bool>(),
             py::arg("elf_path"),
             py::arg("isa") = "rv64gc",
             py::arg("num_instrs") = 200,
             py::arg("verbose") = false,
             R"pbdoc(
             Create a SpikeEngine instance

             Args:
                 elf_path: Path to pre-compiled ELF file with nops
                 isa: ISA string (default: "rv64gc")
                 num_instrs: Number of instructions to generate (default: 200)
                 verbose: Enable verbose output (default: false)
             )pbdoc")

        .def_static("get_instruction_size", &SpikeEngine::get_instruction_size,
             py::arg("machine_code"),
             R"pbdoc(
             Detect instruction size from machine code

             Args:
                 machine_code: 32-bit machine code

             Returns:
                 Instruction size in bytes (2 for compressed, 4 for standard)
             )pbdoc")

        .def("initialize", &SpikeEngine::initialize,
             R"pbdoc(
             Initialize Spike and execute template initialization code
             Returns True on success, False on error (check get_last_error())
             )pbdoc")

        .def("set_checkpoint", &SpikeEngine::set_checkpoint,
             "Save current processor state as checkpoint")

        .def("restore_checkpoint", &SpikeEngine::restore_checkpoint,
             "Restore processor state from last checkpoint")

        .def("execute_sequence", &SpikeEngine::execute_sequence,
             py::arg("machine_codes"),
             py::arg("sizes"),
             py::arg("max_steps") = 10000,
             R"pbdoc(
             Execute a sequence of instructions

             Unified execution method that handles all cases:
             - Single instruction: execute_sequence([code], [size])
             - Forward jump: execute_sequence([jump, middle...], [sizes...])
             - Backward loop: execute_sequence([init, body..., decr, branch], [sizes...])

             Execution logic:
             1. Write all instructions to memory
             2. Calculate end_addr = current_addr + sum(sizes)
             3. Execute until PC >= end_addr
             4. Each step handles traps automatically

             For loops (backward branches):
             - When branch jumps back, PC < end_addr, so execution continues
             - When branch falls through, PC >= end_addr, loop exits

             Args:
                 machine_codes: List of machine codes to execute
                 sizes: List of instruction sizes (2 or 4 bytes each)
                 max_steps: Maximum execution steps (safety limit, default: 10000)

             Returns:
                 Number of steps executed
             )pbdoc")

        //======================================================================
        // Engine Configuration Getters
        //======================================================================

        .def("get_mem_region_start", &SpikeEngine::get_mem_region_start,
             "Get mem_region start address (for testing memory operations)")

        .def("get_mem_region_size", &SpikeEngine::get_mem_region_size,
             "Get mem_region size in bytes")

        .def("get_current_index", &SpikeEngine::get_current_index,
             "Get current instruction index")

        .def("get_num_instrs", &SpikeEngine::get_num_instrs,
             "Get total number of instructions")

        .def("get_last_error", &SpikeEngine::get_last_error,
             "Get last error message")

        //======================================================================
        // Execution State Getters
        //======================================================================

        .def("was_last_execution_trapped", &SpikeEngine::was_last_execution_trapped,
             R"pbdoc(
             Check if the last executed instruction triggered a trap/exception.

             This is useful for logging - instructions that cause traps are handled
             by the exception handler (which skips them), but they are still "accepted"
             from the fuzzer's perspective.

             Returns:
                 True if the last instruction triggered a trap, False otherwise
             )pbdoc")

        .def("get_last_trap_handler_steps", &SpikeEngine::get_last_trap_handler_steps,
             R"pbdoc(
             Get the number of trap handler steps executed in the last execution.

             Returns 0 if no trap occurred.

             Returns:
                 Number of steps executed in trap handler
             )pbdoc")

        //======================================================================
        // Modular Component Accessors
        //======================================================================

        .def("get_state_query",
             static_cast<StateQuery* (SpikeEngine::*)()>(&SpikeEngine::get_state_query),
             py::return_value_policy::reference_internal,
             R"pbdoc(
             Get the StateQuery interface for ALL processor state queries.

             StateQuery is the ONLY interface for querying processor state.
             It provides comprehensive CPU state query interfaces:

             Basic Registers:
               - get_xpr(reg_index) -> uint64: Get integer register value
               - get_all_xpr() -> list[uint64]: Get all integer registers
               - get_fpr(reg_index) -> uint64: Get floating-point register value
               - get_all_fpr() -> list[uint64]: Get all floating-point registers
               - get_pc() -> uint64: Get program counter

             CSRs:
               - get_csr(addr) -> uint64: Get CSR value by address
               - get_all_csrs() -> dict: Get all accessible CSR values

             Memory:
               - read_mem(addr, size) -> bytes: Read memory

             Commit Log:
               - get_commit_log() -> CommitLog: Get last instruction side effects
               - clear_commit_log(): Clear commit log before next instruction

             Privilege State:
               - get_privilege_state() -> PrivilegeState: Get privilege mode info
               - did_privilege_change() -> bool: Check if privilege changed

             Trap/Exception:
               - get_last_trap_info() -> TrapInfo: Get last trap details

             Vector Unit:
               - get_vector_state(include_regfile=False) -> VectorState
               - is_vector_enabled() -> bool

             Load Reservation (LR/SC):
               - get_reservation_state() -> ReservationState
               - has_reservation() -> bool
               - clear_reservation(): Clear reservation

             Debug State:
               - get_debug_state() -> DebugState

             Example:
                 sq = engine.get_state_query()
                 pc = sq.get_pc()
                 x1 = sq.get_xpr(1)
                 mstatus = sq.get_csr(0x300)
                 commit = sq.get_commit_log()

             Returns:
                 StateQuery object (valid for lifetime of SpikeEngine)
             )pbdoc")

        .def("get_checkpoint_manager",
             static_cast<CheckpointManager* (SpikeEngine::*)()>(&SpikeEngine::get_checkpoint_manager),
             py::return_value_policy::reference_internal,
             R"pbdoc(
             Get the CheckpointManager for advanced checkpoint operations.

             Returns:
                 CheckpointManager object (valid for lifetime of SpikeEngine)
             )pbdoc");

    // Version info
    m.attr("__version__") = "4.0.0";
}

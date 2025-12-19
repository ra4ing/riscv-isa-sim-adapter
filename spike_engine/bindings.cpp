// Copyright (c) 2024-2025 DiveFuzz Project
// SPDX-License-Identifier: Mulan PSL v2

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "spike_engine.h"

namespace py = pybind11;
using namespace spike_engine;

PYBIND11_MODULE(spike_engine, m) {
    m.doc() = "Efficient Spike execution engine with checkpointing for DiveFuzz";

    // Constants
    m.attr("IMMEDIATE_NOT_PRESENT") = IMMEDIATE_NOT_PRESENT;

    // ExecutionResult class
    py::class_<ExecutionResult>(m, "ExecutionResult")
        .def(py::init<>())
        .def(py::init<const std::vector<uint64_t>&, const std::vector<uint64_t>&>(),
             py::arg("source_values_before"),
             py::arg("dest_values_after"))
        .def_readwrite("source_values_before", &ExecutionResult::source_values_before,
             "Source register values captured BEFORE execution (for XOR computation)")
        .def_readwrite("dest_values_after", &ExecutionResult::dest_values_after,
             "Destination register values captured AFTER execution (for bug filtering)");

    // Checkpoint class
    py::class_<Checkpoint>(m, "Checkpoint")
        .def(py::init<>())
        .def_readwrite("xpr", &Checkpoint::xpr, "General-purpose registers (x0-x31)")
        .def_readwrite("fpr", &Checkpoint::fpr, "Floating-point registers (f0-f31)")
        .def_readwrite("pc", &Checkpoint::pc, "Program counter")
        .def_readwrite("instr_index", &Checkpoint::instr_index, "Current instruction index");

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

        .def("execute_instruction", &SpikeEngine::execute_instruction,
             py::arg("machine_code"),
             py::arg("source_regs"),
             py::arg("dest_regs"),
             py::arg("immediate") = 0,
             R"pbdoc(
             Execute one instruction and return register values

             Args:
                 machine_code: 32-bit machine code
                 source_regs: List of source register indices (read before execution)
                 dest_regs: List of destination register indices (read after execution)
                 immediate: Immediate value (default: 0)

             Returns:
                 ExecutionResult with:
                 - source_values_before: Source register values before execution (for XOR)
                 - dest_values_after: Destination register values after execution (for bug filtering)
             )pbdoc")

        .def("get_xpr", &SpikeEngine::get_xpr,
             py::arg("reg_index"),
             "Get general-purpose register value (x0-x31)")

        .def("get_fpr", &SpikeEngine::get_fpr,
             py::arg("reg_index"),
             "Get floating-point register value (f0-f31)")

        .def("get_pc", &SpikeEngine::get_pc,
             "Get program counter value")

        .def("get_current_index", &SpikeEngine::get_current_index,
             "Get current instruction index")

        .def("get_num_instrs", &SpikeEngine::get_num_instrs,
             "Get total number of instructions")

        .def("get_last_error", &SpikeEngine::get_last_error,
             "Get last error message");

    // Version info
    m.attr("__version__") = "2.0.0";
}

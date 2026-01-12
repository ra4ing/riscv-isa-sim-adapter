# Makefile for Spike Engine with Checkpointing
#
# Usage:
#   make          - Build the Python module
#   make clean    - Clean build artifacts
#   make test     - Run tests
#   make help     - Show help information

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -fPIC -O2 -g -Wall -Wextra -Wno-unused-parameter

# Directories
SPIKE_ROOT = ..
SPIKE_BUILD_DIR = $(SPIKE_ROOT)/build
CURRENT_DIR = $(shell pwd)

# Python configuration
# Use 'python' instead of 'python3' to match DiveFuzz generator runtime (Python 3.11)
PYTHON = python
PYTHON_VERSION = $(shell $(PYTHON) -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

# Try to find the correct python config
PYTHON_CONFIG = $(shell \
	if command -v python$(PYTHON_VERSION)-config >/dev/null 2>&1; then \
		echo "python$(PYTHON_VERSION)-config"; \
	else \
		echo "python3-config"; \
	fi)

# Python build parameters
# Use manually installed pybind11 if Python module not available
PYBIND11_INCLUDE_PYTHON = $(shell $(PYTHON) -c "import pybind11; print(pybind11.get_include())" 2>/dev/null)
PYBIND11_INCLUDE = $(if $(PYBIND11_INCLUDE_PYTHON),$(PYBIND11_INCLUDE_PYTHON),/usr/local/include)
PYTHON_INCLUDES = $(shell $(PYTHON_CONFIG) --includes 2>/dev/null || $(PYTHON) -c "import sysconfig; print('-I' + sysconfig.get_path('include'))")
PYTHON_LDFLAGS = $(shell $(PYTHON_CONFIG) --ldflags 2>/dev/null || echo "-lpython3")
PYTHON_EXT_SUFFIX_CONFIG = $(shell $(PYTHON_CONFIG) --extension-suffix 2>/dev/null)
PYTHON_EXT_SUFFIX = $(if $(PYTHON_EXT_SUFFIX_CONFIG),$(PYTHON_EXT_SUFFIX_CONFIG),$(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"))

# Include directories
INCLUDES = -I$(SPIKE_ROOT) \
           -I$(SPIKE_BUILD_DIR) \
           -I$(SPIKE_ROOT)/riscv \
           -I$(SPIKE_ROOT)/fesvr \
           -I$(SPIKE_ROOT)/softfloat \
           -I$(SPIKE_ROOT)/disasm \
           -I$(SPIKE_ROOT)/fdt

# Spike libraries
SPIKE_LIBS = $(SPIKE_BUILD_DIR)/libriscv.a \
             $(SPIKE_BUILD_DIR)/libdisasm.a \
             $(SPIKE_BUILD_DIR)/libsoftfloat.a \
             $(SPIKE_BUILD_DIR)/libfesvr.a \
             $(SPIKE_BUILD_DIR)/libfdt.a

# System libraries
SYS_LIBS = -lpthread -ldl -lboost_regex -lboost_system

# Source files
ENGINE_SOURCES = spike_engine.cpp
STATE_QUERY_SOURCES = state_query.cpp
CHECKPOINT_SOURCES = checkpoint.cpp
BINDING_SOURCES = bindings.cpp

# Object files
ENGINE_OBJECTS = $(ENGINE_SOURCES:.cpp=.o)
STATE_QUERY_OBJECTS = $(STATE_QUERY_SOURCES:.cpp=.o)
CHECKPOINT_OBJECTS = $(CHECKPOINT_SOURCES:.cpp=.o)
BINDING_OBJECTS = $(BINDING_SOURCES:.cpp=.o)

# Target files
PYTHON_MODULE = spike_engine$(PYTHON_EXT_SUFFIX)

# Default target
.PHONY: all
all: check-deps $(PYTHON_MODULE)

# Check dependencies
.PHONY: check-deps
check-deps:
	@echo "Checking dependencies..."
	@if [ ! -f "$(SPIKE_BUILD_DIR)/libriscv.a" ]; then \
		echo "Error: Spike not built. Please run:"; \
		echo "  cd $(SPIKE_ROOT) && mkdir -p build && cd build && ../configure && make"; \
		exit 1; \
	fi
	@if ! $(PYTHON) -c "import pybind11" >/dev/null 2>&1; then \
		if [ ! -d "/usr/local/include/pybind11" ]; then \
			echo "Error: pybind11 not found. Please install it:"; \
			echo "  pip install pybind11"; \
			echo "Or install headers manually (see INSTALL.md)"; \
			exit 1; \
		else \
			echo "Note: Using pybind11 headers from /usr/local/include"; \
		fi \
	fi
	@if [ -z "$(PYBIND11_INCLUDE)" ]; then \
		echo "Error: Failed to get pybind11 include path"; \
		exit 1; \
	fi
	@if [ -z "$(PYTHON_EXT_SUFFIX)" ]; then \
		echo "Error: Failed to get Python extension suffix"; \
		exit 1; \
	fi
	@echo "Dependencies check passed."
	@echo "Python version: $(PYTHON_VERSION)"
	@echo "Python config: $(PYTHON_CONFIG)"
	@echo "Extension suffix: $(PYTHON_EXT_SUFFIX)"
	@echo "Pybind11 include: $(PYBIND11_INCLUDE)"

# Build engine object
$(ENGINE_OBJECTS): $(ENGINE_SOURCES) spike_engine.h state_query.h checkpoint.h
	@echo "Compiling spike_engine..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(PYTHON_INCLUDES) -c $< -o $@

# Build state query object (depends on checkpoint.h for ReservationState)
$(STATE_QUERY_OBJECTS): $(STATE_QUERY_SOURCES) state_query.h checkpoint.h
	@echo "Compiling state_query..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(PYTHON_INCLUDES) -c $< -o $@

# Build checkpoint object
$(CHECKPOINT_OBJECTS): $(CHECKPOINT_SOURCES) checkpoint.h
	@echo "Compiling checkpoint..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(PYTHON_INCLUDES) -c $< -o $@

# Build Python binding object
$(BINDING_OBJECTS): $(BINDING_SOURCES) spike_engine.h state_query.h checkpoint.h
	@echo "Compiling Python bindings..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -I$(PYBIND11_INCLUDE) $(PYTHON_INCLUDES) -c $< -o $@

# Build Python module
$(PYTHON_MODULE): $(ENGINE_OBJECTS) $(STATE_QUERY_OBJECTS) $(CHECKPOINT_OBJECTS) $(BINDING_OBJECTS)
	@echo "Linking Python module..."
	$(CXX) -shared $(PYTHON_LDFLAGS) -o $@ $^ $(SPIKE_LIBS) $(SYS_LIBS)
	@echo ""
	@echo "=========================================="
	@echo "Build completed successfully!"
	@echo "Python module: $(CURRENT_DIR)/$(PYTHON_MODULE)"
	@echo "=========================================="
	@echo ""
	@echo "To use in Python:"
	@echo "  import sys"
	@echo "  sys.path.insert(0, '$(CURRENT_DIR)')"
	@echo "  import spike_engine"
	@echo ""

# Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	rm -f *.o
	rm -f spike_engine*.so
	rm -f spike_engine*.dylib
	rm -f *.pyc
	rm -rf __pycache__
	rm -rf build/
	@echo "Clean completed."

# Run tests
.PHONY: test
test: $(PYTHON_MODULE)
	@echo "Running tests..."
	@if [ -f test_spike_engine.py ]; then \
		$(PYTHON) test_spike_engine.py; \
	else \
		echo "test_spike_engine.py not found"; \
	fi

# Install module (copy to a standard location)
.PHONY: install
install: $(PYTHON_MODULE)
	@echo "Installing spike_engine module..."
	@SITE_PACKAGES=$$($(PYTHON) -c "import site; print(site.getsitepackages()[0])"); \
	echo "Installing to: $$SITE_PACKAGES"; \
	cp $(PYTHON_MODULE) $$SITE_PACKAGES/
	@echo "Installation completed."

# Help information
.PHONY: help
help:
	@echo "Spike Engine Makefile"
	@echo "====================="
	@echo ""
	@echo "Available targets:"
	@echo "  all (default) - Build the Python module"
	@echo "  clean         - Clean build artifacts"
	@echo "  test          - Run tests"
	@echo "  install       - Install module to Python site-packages"
	@echo "  help          - Show this help message"
	@echo ""
	@echo "Prerequisites:"
	@echo "  1. Build Spike first:"
	@echo "     cd $(SPIKE_ROOT) && mkdir -p build && cd build"
	@echo "     ../configure && make"
	@echo ""
	@echo "  2. Install Python dependencies:"
	@echo "     pip install pybind11"
	@echo ""

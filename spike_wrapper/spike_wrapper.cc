#include "spike_wrapper.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <unistd.h>
#include <cstring>
#include <sys/wait.h>
#include <fcntl.h>
#include <stdexcept>
#include <sstream>
#include <memory>
#include <fstream>
#include <poll.h>
#include <errno.h>
#include <signal.h>
#include <time.h>

// Global error information storage
static std::string g_last_error;

// Hard wall-clock timeout for one spike debug query. A derailing candidate
// instruction can make the "until <magic value>" debug command never
// terminate; without this guard the query (and its calling worker) hangs
// forever burning one core.
static const double kSpikeQueryTimeoutSec = 2.0;

static double monotonic_now() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

// Execute cmd via /bin/sh with stdout+stderr captured, enforcing a wall-clock
// timeout: on expiry the whole child process group (sh + spike) is SIGKILLed
// and the query raises, which the caller already treats as a rejected
// candidate. Fast queries are byte-for-byte identical to plain popen().
static std::string run_with_timeout(const std::string& cmd, double timeout_sec) {
    int fds[2];
    if (pipe(fds) != 0)
        throw std::runtime_error("pipe() failed");

    pid_t pid = fork();
    if (pid < 0) {
        close(fds[0]);
        close(fds[1]);
        throw std::runtime_error("fork() failed");
    }
    if (pid == 0) {
        setpgid(0, 0); // own group so the timeout kills sh+spike together
        close(fds[0]);
        dup2(fds[1], STDOUT_FILENO);
        dup2(fds[1], STDERR_FILENO);
        close(fds[1]);
        execl("/bin/sh", "sh", "-c", cmd.c_str(), (char*)nullptr);
        _exit(127);
    }
    close(fds[1]);
    setpgid(pid, pid); // parent side too; child may already have done it

    std::string result;
    char buffer[4096];
    const double deadline = monotonic_now() + timeout_sec;
    bool timed_out = false;

    struct pollfd pfd;
    pfd.fd = fds[0];
    pfd.events = POLLIN;
    for (;;) {
        double remaining = deadline - monotonic_now();
        if (remaining <= 0) { timed_out = true; break; }
        int r = poll(&pfd, 1, (int)(remaining * 1000.0));
        if (r < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (r == 0) { timed_out = true; break; }
        if (pfd.revents & (POLLIN | POLLHUP | POLLERR)) {
            ssize_t n = read(fds[0], buffer, sizeof(buffer));
            if (n <= 0) break; // EOF: child side closed
            result.append(buffer, (size_t)n);
            if (pfd.revents & (POLLHUP | POLLERR)) {
                while ((n = read(fds[0], buffer, sizeof(buffer))) > 0)
                    result.append(buffer, (size_t)n);
                break;
            }
        }
    }

    if (timed_out)
        kill(-pid, SIGKILL);
    close(fds[0]);
    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}

    if (timed_out)
        throw std::runtime_error("spike query timed out (derailed candidate?)");
    return result;
}

// Run Spike debugging from a file path 
// – use --debug-cmd-from-string to avoid creating temporary files for debug commands.
// Run Spike debugging from a file path
// – use --debug-cmd-from-string to avoid creating temporary files for debug commands.
static std::string run_spike_debug_cmd_str_elf_file(const std::string& elf_file_path,
                                           const std::string& debug_cmds_string,
                                           const std::string& isa_string) {
    char spike_cmd[4096];
    snprintf(spike_cmd, sizeof(spike_cmd),
            "spike -d --isa=%s --debug-cmd-from-string='%s' %s 2>&1",
            isa_string.c_str(), debug_cmds_string.c_str(), elf_file_path.c_str());
    return run_with_timeout(spike_cmd, kSpikeQueryTimeoutSec);
}

static std::string run_spike_debug_cmd_file_elf_file(const std::string& elf_file_path,
                                           const std::string& debug_cmds_path,
                                           const std::string& isa_string) {
    char spike_cmd[4096];
    snprintf(spike_cmd, sizeof(spike_cmd),
            "spike -d --isa=%s --debug-cmd='%s' %s 2>&1",
            isa_string.c_str(), debug_cmds_path.c_str(), elf_file_path.c_str());
    return run_with_timeout(spike_cmd, kSpikeQueryTimeoutSec);
}



// C API implementation
extern "C" {

int spike_debug_cmd_str_elf_file(const char* elf_file_path,
                         const char* debug_cmds_string,
                         const char* isa_string,
                         char* output_buffer,
                         size_t buffer_size) {
    try {
        if (!elf_file_path || !debug_cmds_string || !isa_string || !output_buffer) {
            g_last_error = "Invalid parameters";
            return -1;
        }

        std::string result = run_spike_debug_cmd_str_elf_file(
            std::string(elf_file_path),
            std::string(debug_cmds_string),
            std::string(isa_string)
        );

        size_t copy_size = std::min(result.length(), buffer_size - 1);
        std::memcpy(output_buffer, result.c_str(), copy_size);
        output_buffer[copy_size] = '\0';

        return static_cast<int>(copy_size);

    } catch (const std::exception& e) {
        g_last_error = e.what();
        return -1;
    }
}

int spike_debug_cmd_file_elf_file(const char* elf_file_path,
                         const char* debug_cmds_path,
                         const char* isa_string,
                         char* output_buffer,
                         size_t buffer_size) {
    try {
        if (!elf_file_path || !debug_cmds_path || !isa_string || !output_buffer) {
            g_last_error = "Invalid parameters";
            return -1;
        }

        std::string result = run_spike_debug_cmd_file_elf_file(
            std::string(elf_file_path),
            std::string(debug_cmds_path),
            std::string(isa_string)
        );

        size_t copy_size = std::min(result.length(), buffer_size - 1);
        std::memcpy(output_buffer, result.c_str(), copy_size);
        output_buffer[copy_size] = '\0';

        return static_cast<int>(copy_size);

    } catch (const std::exception& e) {
        g_last_error = e.what();
        return -1;
    }
}



const char* spike_get_last_error(void) {
    return g_last_error.c_str();
}

}

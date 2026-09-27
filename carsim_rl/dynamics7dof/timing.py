"""Collector-only, read-only observation of the VS internal clock.

No DLL is loaded here. Bind only AFTER vs_read_configuration, discard BEFORE
the solver is terminated, and never write through the pointer. T is a solver
clock observation, not a proof that every exported force/state has that time.

CarSim 2022.1 local documentation:
  Help/Manuals/vs_commands.pdf p14: T is current simulation time, read-only.
  Help/Memos/VS_Commands_API.pdf p26: vs_get_var_ptr accesses model variables.
  Help/Manuals/system_parameters.pdf pp17-18,27: AM/RK half steps / IO options.
"""

import ctypes
import math


TIME_SOURCE = "vs_get_var_ptr(T)"


class SolverClock:
    def __init__(self):
        self.pointer = None
        self.reason = "not_bound"

    def bind(self, dll, expected_start):
        try:
            getter = dll.vs_get_var_ptr
            getter.argtypes = [ctypes.c_char_p]
            getter.restype = ctypes.POINTER(ctypes.c_double)
            pointer = getter(b"T")
            if not pointer:
                raise ValueError("T returned a null pointer")
            value = float(pointer[0])
            if not math.isfinite(value) or not math.isclose(value, expected_start, abs_tol=1e-8, rel_tol=0):
                raise ValueError(f"Initial T={value} differs from configured start {expected_start}")
            self.pointer = pointer
            self.reason = "available"
        except (AttributeError, TypeError, ValueError, OSError) as exc:
            self.pointer = None
            self.reason = f"unavailable: {type(exc).__name__}: {exc}"
        return self.read()

    def read(self):
        if self.pointer is None:
            return None
        try:
            value = float(self.pointer[0])
            if not math.isfinite(value):
                raise ValueError("Non-finite T")
            return value
        except (ValueError, OSError) as exc:
            self.pointer = None
            self.reason = f"read_failed: {exc}"
            return None

    def discard(self):
        self.pointer = None


class ObservedSolver:
    """Read the clock around integration, BEFORE CarSimEnv can terminate the DLL.

    Forward exactly the same arguments, result and termination calls. The
    original vs_solver class and every non-collector entry point stay unchanged.
    """

    def __init__(self, solver, clock):
        self.solver, self.clock = solver, clock
        self.before = self.after = None

    def __getattr__(self, name):
        return getattr(self.solver, name)

    def integrate_io_inplace(self, *args, **kwargs):
        self.before = self.clock.read()
        self.after = None
        result = self.solver.integrate_io_inplace(*args, **kwargs)
        self.after = self.clock.read()
        return result

    def terminate_run(self, *args, **kwargs):
        self.clock.discard()
        return self.solver.terminate_run(*args, **kwargs)

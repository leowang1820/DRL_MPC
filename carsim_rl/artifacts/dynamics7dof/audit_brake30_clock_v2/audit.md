# CarSim capture audit

Source: E:\modelcodeE\DRL_MPC\DRL_MPC\carsim_rl\artifacts\dynamics7dof\capture_20260927_160434_220127
Data integrity: PASS
Rows: 17701
Ready for residual training: NO

Diagnostic time basis: wrapper_nominal_unverified
Actual solver clock rows: 0
Lateral error: RMS 0.148485 m, max abs 0.343926 m
Forward braking rows: 1100
FL brake moment: -619.522741 .. -0.002830 N*m; negative fraction during forward braking: 1.0
FR brake moment: -619.522741 .. -0.002830 N*m; negative fraction during forward braking: 1.0
RL brake moment: -232.008528 .. -0.001589 N*m; negative fraction during forward braking: 1.0
RR brake moment: -232.008528 .. -0.001589 N*m; negative fraction during forward braking: 1.0
AX planar consistency RMS difference: 0.0013023912489216844 m/s^2 (NOT 7DOF model error)
AY planar consistency RMS difference: 0.003538884756540206 m/s^2 (NOT 7DOF model error)

## Errors

None.

## Warnings

- Legacy capture has no actual solver time; no 0.5 ms shift has been inferred or applied
- AM/RK uses internal half steps; inspect OPT_IO_UPDATE. OPT_IO_SYNC_FM alone does not establish timing
- Tracking metrics use current path file; historical path identity is unverified for this legacy capture

## Pending physical checks

- Per-export state/force timestamp alignment is not yet physically verified.
- Signed drive + brake is only a candidate net torque; wheel torque balance remains unverified.
- CarSim Ax/Ay are not automatically the reference model's LoadMemory.

# CarSim capture audit

Source: E:\modelcodeE\DRL_MPC\DRL_MPC\carsim_rl\artifacts\dynamics7dof\capture_20260927_155420_045453
Data integrity: PASS
Rows: 13601
Ready for residual training: NO

Diagnostic time basis: wrapper_nominal_unverified
Actual solver clock rows: 0
Lateral error: RMS 0.132128 m, max abs 0.328971 m
Forward braking rows: 0
FL brake moment: -0.500000 .. -0.500000 N*m; negative fraction during forward braking: None
FR brake moment: -0.500000 .. -0.500000 N*m; negative fraction during forward braking: None
RL brake moment: -0.500000 .. -0.500000 N*m; negative fraction during forward braking: None
RR brake moment: -0.500000 .. -0.500000 N*m; negative fraction during forward braking: None
AX planar consistency RMS difference: 0.002081801689833315 m/s^2 (NOT 7DOF model error)
AY planar consistency RMS difference: 0.005578523138749871 m/s^2 (NOT 7DOF model error)

## Errors

None.

## Warnings

- Legacy capture has no actual solver time; no 0.5 ms shift has been inferred or applied
- AM/RK uses internal half steps; inspect OPT_IO_UPDATE. OPT_IO_SYNC_FM alone does not establish timing
- No active forward braking samples; brake sign cannot be established from this episode
- Tracking metrics use current path file; historical path identity is unverified for this legacy capture

## Pending physical checks

- Per-export state/force timestamp alignment is not yet physically verified.
- Signed drive + brake is only a candidate net torque; wheel torque balance remains unverified.
- CarSim Ax/Ay are not automatically the reference model's LoadMemory.

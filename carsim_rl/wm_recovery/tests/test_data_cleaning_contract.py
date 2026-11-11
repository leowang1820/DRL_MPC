# -*- coding: utf-8 -*-
"""验证清洗候选能够通过项目原有的严格 CandidateEpisode.load()。"""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from wm_recovery.carsim_candidate import _fingerprint
from wm_recovery.carsim_experimental import _candidate_facts
from wm_recovery.offline import CandidateEpisode
from wm_recovery.data_cleaning import (
    QualityPolicy,
    _candidate_report_after_trim,
    _trim_arrays,
)


def strict_candidate_arrays(n=600):
    time = np.arange(n + 1) * 0.01
    state = np.zeros((n+1,7),dtype=float)
    state[:,0] = 40/3.6
    return {
        "time_s": time,
        "state_si": state,
        "command_si": np.zeros((n,5)),
        "nominal_7dof_input_si": np.zeros((n,6)),
        "measured_steer_rad": np.zeros((n+1,2)),
        "reported_external_torque_nm": np.zeros((n+1,4)),
        "reported_accel_mps2": np.zeros((n+1,2)),
        "import_native": np.zeros((n,6)),
        "my_dr_native": np.zeros((n+1,4)),
        "nominal_initial_load_memory_mps2": np.zeros(2),
    }


def strict_original_report(arrays):
    n=len(arrays["command_si"])
    return {
        "format": "wm_recovery_carsim_candidate_v1",
        "status": "candidate_created_not_training_ready",
        "source_kind": "carsim_env",
        "raw_source": "raw_capture.npz",
        "raw_source_sha256": "strict_fixture_raw_sha_not_a_real_run",
        "canonical_physical_arrays_sha256": _fingerprint(arrays),
        "samples": n+1,
        "intervals": n,
        "dt_s": 0.01,
        "actual_mu": "user_confirmed_nominal_0.8_not_rechecked",
        "vehicle_parameter_alignment": "user_confirmed_not_rechecked",
        "split": "unassigned",
        "model_input_definition":
            "equal_front_commanded_steer_plus_four_commanded_external_torques",
        "interval_alignment":
            "command_i_holds_from_t_i_to_t_i_plus_1",
        "torque_echo_matches": True,
        "max_torque_echo_error_nm": 0.0,
        "ready_for_wm_training": False,
    }


class StrictContractTests(unittest.TestCase):

    def test_cleaned_candidate_loads_with_real_loader(self):
        original_arrays = strict_candidate_arrays()
        original_report = strict_original_report(original_arrays)
        cleaned_arrays = _trim_arrays(original_arrays, keep=295)
        cleaned_report = _candidate_report_after_trim(
            original_report, cleaned_arrays, 295,
            original_report["canonical_physical_arrays_sha256"],
            QualityPolicy(),
        )

        self.assertTrue(cleaned_report["quality_cleaned"])
        self.assertEqual(
            cleaned_report["status"],"candidate_created_not_training_ready"
        )
        self.assertIs(cleaned_report["ready_for_wm_training"],False)
        self.assertEqual(cleaned_report["samples"],296)
        self.assertEqual(cleaned_report["intervals"],295)

        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            np.savez_compressed(p/"candidate.npz",**cleaned_arrays)
            (p/"candidate_report.json").write_text(
                json.dumps(cleaned_report,ensure_ascii=False),
                encoding="utf-8",
            )

            # 不替换、不绕过原读取器：保证用户遇到的真实校验能够通过。
            ep=CandidateEpisode.load(p)
            self.assertEqual(ep.state.shape,(296,7))
            self.assertEqual(ep.nominal_input.shape,(295,6))
            self.assertEqual(ep.command.shape,(295,5))
            self.assertEqual(
                ep.source_sha256,
                cleaned_report["canonical_physical_arrays_sha256"]
            )

            # 再使用训练入口的现有严格候选校验验证一次。
            facts=_candidate_facts(p)
            self.assertEqual(
                facts["candidate_sha256"],
                cleaned_report["canonical_physical_arrays_sha256"]
            )

            # 反例：改成之前的错误 status，原加载器必须继续拒绝。
            invalid=dict(cleaned_report)
            invalid["status"]="candidate_cleaned_not_physically_validated"
            (p/"candidate_report.json").write_text(
                json.dumps(invalid,ensure_ascii=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError,"状态与只读数据契约"):
                CandidateEpisode.load(p)


if __name__ == "__main__":
    unittest.main()

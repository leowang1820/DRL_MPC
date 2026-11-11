# -*- coding: utf-8 -*-
"""多速度控制入口无 CarSim DLL 的回归测试。"""

from collections import Counter
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from wm_recovery import multi_speed_runner as m


def create_fake_sim(root, speed, run_name=None):
    """创建可模拟宏展开的 SIM 和参数文件，不是真实 CarSim 文件。"""
    run_name = run_name or f"Run_{speed}"
    root = Path(root)
    folder = root / f"{speed}kmh"
    folder.mkdir(exist_ok=True, parents=True)
    results = root / "Results" / run_name
    results.mkdir(parents=True, exist_ok=True)
    (results / "Run_all.par").write_text(
        f"TEST_PARAM SPEED={speed} RUN={run_name}",
        encoding="utf-8",
    )

    sim = folder / "simfile.sim"
    sim.write_text(
        "SIMFILE\n"
        f"SET_MACRO $(ROOT_FILE_NAME)$ {run_name}\n"
        f"SET_MACRO $(WORK_DIR)$ {root.as_posix()}/\n"
        "SET_MACRO $(OUT)$ $(WORK_DIR)$Results/$(ROOT_FILE_NAME)$/LastRun\n"
        "INPUT $(WORK_DIR)$Results/$(ROOT_FILE_NAME)$/Run_all.par\n"
        "FILEBASE $(OUT)$\n"
        "ERDFILE $(OUT)$.vs\n"
        "PORTS_IMP 1,6\n"
        "PORTS_EXP 1,61\n"
        "EXT_MODEL_STEP 0.001\n"
        "END\n",
        encoding="utf-8",
    )
    return sim


class FakeBridge:
    actual = {}

    def __init__(self, sim_path, prediction_dt_s):
        name = Path(sim_path).parent.name
        speed = int(name[:-3])
        self.latest = types.SimpleNamespace(
            data=types.SimpleNamespace(
                Vx=self.actual.get(speed, float(speed))
            )
        )
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, etype, exc, trace):
        self.closed = True


class MultiSpeedTests(unittest.TestCase):

    def test_find_eight_speed_directories_and_expand_macros(self):
        with tempfile.TemporaryDirectory() as tmp:
            for speed in m.EXPECTED_SPEEDS:
                create_fake_sim(tmp, speed)
            paths = m.discover(tmp, m.EXPECTED_SPEEDS)
            report = m.audit_records(paths, strict_interface=False)
            self.assertEqual(report["errors"], [])
            self.assertEqual(len(report["entries"]), 8)
            self.assertEqual(len({
                r["input_par_path"] for r in report["entries"]
            }), 8)
            self.assertEqual(len({
                r["filebase"] for r in report["entries"]
            }), 8)
            self.assertTrue(all(r["par_exists"]
                                for r in report["entries"]))
            self.assertTrue(all(
                r["actual_initial_speed_kmh"] is None
                for r in report["entries"]
            ))

    def test_collisions_of_shared_input_and_output_are_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = create_fake_sim(tmp, 40, run_name="Run_same")
            b = create_fake_sim(tmp, 60, run_name="Run_same")
            result = m.audit_records({40: a, 60: b},
                                     strict_interface=False)
            errors = "\n".join(result["errors"])
            self.assertIn("共用同一个 INPUT", errors)
            self.assertIn("共用同一个 FILEBASE", errors)
            with self.assertRaises(RuntimeError):
                m.require_audit_good(result)

    def test_wrong_initial_speed_cannot_pass_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = {
                40: create_fake_sim(tmp, 40),
                60: create_fake_sim(tmp, 60),
            }
            audit = m.audit_records(paths, strict_interface=False)
            with patch.dict(FakeBridge.actual, {60: 79.0}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "初始车速不匹配"):
                    m.probe_initial_speeds(
                        paths, audit, tolerance_kmh=1.0,
                        bridge_factory=FakeBridge
                    )

    def test_correct_initial_speed_passes_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = {
                40: create_fake_sim(tmp, 40),
                60: create_fake_sim(tmp, 60),
            }
            audit = m.audit_records(paths, strict_interface=False)
            with patch.dict(FakeBridge.actual, {}, clear=True):
                measurements = m.probe_initial_speeds(
                    paths, audit, bridge_factory=FakeBridge
                )
            self.assertEqual(measurements, {40:40.0,60:60.0})

    def test_high_speed_opt_in_before_solver(self):
        with self.assertRaisesRegex(ValueError, "allow-high-speed"):
            m.capture_speeds(
                "not_a_real_folder", "not_real_output",
                speeds=[120], allow_high_speed=False
            )

    def test_existing_output_prevented_by_non_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "multi"
            p.mkdir()
            with patch.object(
                m, "_audit_and_probe",
                return_value=(
                    {40: Path("mock.sim")},
                    {"entries":[{
                        "expected_speed_kmh":40,
                        "sim_sha256":"simhash",
                        "parameter_sha256":"parhash",
                        "input_par_path":"par",
                        "filebase":"filebase",
                    }]},
                    {40:40.0}
                )
            ):
                with self.assertRaises(FileExistsError):
                    m.capture_speeds(
                        tmp, p, [40], allow_high_speed=False
                    )

    def test_selected_speed_option(self):
        self.assertEqual(m.selected_speeds(["40","80"]), (40,80))
        self.assertEqual(m.selected_speeds(["all"]),
                         m.EXPECTED_SPEEDS)
        with self.assertRaises(ValueError):
            m.selected_speeds(["80","80"])

    def _setup_fake_captures(self, root, speeds):
        for speed in speeds:
            folder = root / f"{speed}kmh"
            workspace = folder / "workspace"
            workspace.mkdir(parents=True)
            rows = []
            for j, scene in enumerate(
                __import__("wm_recovery.scenario_config",
                           fromlist=["SCENARIOS"]).SCENARIOS
            ):
                name = scene.name
                cand = folder / f"candidate_{name}"
                cand.mkdir()
                (cand / "candidate_report.json").write_text(
                    "{}", encoding="utf-8"
                )
                ep = workspace / "episodes" / name
                ep.mkdir(parents=True)
                meta = {
                    "format":"wm_recovery_experimental_episode_manifest_v1",
                    "split":scene.split,
                    "condition_id":scene.condition_id,
                    "trajectory_id":"original_"+name,
                    "candidate_path":str(cand.resolve()),
                    "candidate_sha256":f"c_{speed}_{j}",
                    "raw_source_sha256":f"r_{speed}_{j}",
                }
                (ep / "experimental_episode.json").write_text(
                    json.dumps(meta,ensure_ascii=False),
                    encoding="utf-8",
                )
                rows.append({
                    "name":name,"episode_path":str(ep.resolve())
                })
            (workspace / "registry.json").write_text(
                json.dumps({
                    "format":m.REGISTRY_VERSION,
                    "episode_count":len(rows),
                    "episodes":rows,
                }),encoding="utf-8"
            )
            (folder / "coverage_report.json").write_text(
                '{"passed":true}',encoding="utf-8"
            )
            (folder / "speed_capture_manifest.json").write_text(
                json.dumps({
                    "format":m.MANIFEST_VERSION,
                    "expected_speed_kmh":speed,
                    "measured_initial_speed_kmh":float(speed),
                    "sim_path":f"mock_{speed}.sim",
                    "sim_sha256":f"shasim_{speed}",
                    "parameter_sha256":f"shapar_{speed}",
                }),encoding="utf-8"
            )

    def _patch_candidates(self):
        def fake_facts(path):
            path = Path(path)
            folder = path.parent
            speed = int(folder.name[:-3])
            name = path.name.removeprefix("candidate_")
            from wm_recovery.scenario_config import SCENARIOS
            idx = [s.name for s in SCENARIOS].index(name)
            return {
                "candidate_sha256":f"c_{speed}_{idx}",
                "raw_source_sha256":f"r_{speed}_{idx}",
                "candidate":types.SimpleNamespace(metadata={
                    "capture_sim_sha256":f"shasim_{speed}",
                    "capture_parameter_sha256":f"shapar_{speed}",
                }),
            }

        mod=types.ModuleType("wm_recovery.carsim_experimental")
        mod._candidate_facts=fake_facts
        mod.MANIFEST_FORMAT=(
            "wm_recovery_experimental_episode_manifest_v1"
        )
        return mod

    def test_merge_all_speeds_and_holdout_partition(self):
        from wm_recovery.scenario_config import SCENARIOS
        source_mod=self._patch_candidates()
        offline=types.ModuleType("wm_recovery.offline")
        offline.NOMINAL_MU=0.8
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"captures"
            root.mkdir()
            self._setup_fake_captures(root,(40,60,80,100))
            with patch.dict(sys.modules, {
                "wm_recovery.carsim_experimental":source_mod,
                "wm_recovery.offline":offline,
            }):
                merged = m.merge_workspaces(
                    root, Path(tmp)/"all_seen", (40,60,80,100)
                )
                rows=json.loads(
                    (merged/"registry.json").read_text(encoding="utf-8")
                )["episodes"]
                self.assertEqual(Counter(r["split"] for r in rows),
                                 {"train":48,"validation":12,"test":12})
                self.assertTrue(
                    (merged/"multispeed_coverage_summary.json").is_file()
                )
                self.assertEqual(
                    len({r["condition_id"] for r in rows}),
                    len(rows)
                )

                held = m.merge_workspaces(
                    root, Path(tmp)/"holdout", (40,60,80,100),
                    validation_speed=80, test_speed=100
                )
                r2=json.loads(
                    (held/"registry.json").read_text(encoding="utf-8")
                )
                self.assertEqual(
                    r2["partition_counts"],
                    {"train":24,"validation":3,"test":3}
                )
                self.assertEqual(
                    r2["speed_split_policy"],"speed_holdout"
                )

    def test_train_blocks_insufficient_coverage_without_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"workspace"
            p.mkdir()
            (p/"multispeed_coverage_summary.json").write_text(
                '{"passed":false}',encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "覆盖率"):
                m.train_merged(p,Path(tmp)/"model",10)


    def test_capture_checks_each_trajectory_initial_speed_and_resume(self):
        """模拟完整 18 轨迹采集并检查首帧车速；不使用真实 Solver。"""
        import numpy as np
        from wm_recovery.scenario_config import SCENARIOS

        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "batch"
            fake_sim = Path(temp) / "40kmh" / "simfile.sim"
            fake_sim.parent.mkdir()
            fake_sim.write_text("SIMFILE", encoding="utf-8")
            audit = {
                "entries":[{
                    "expected_speed_kmh":40,
                    "sim_sha256":"s40",
                    "parameter_sha256":"p40",
                    "input_par_path":"Run_all.par",
                    "filebase":"Run_40/LastRun",
                }],
                "errors":[],
            }
            state = {"calls":0, "wrong_speed":False}
            def fake_capture_all(sim_path, speed_dir):
                state["calls"] += 1
                speed_dir = Path(speed_dir)
                workspace = speed_dir / "workspace"
                workspace.mkdir(parents=True)
                rows = []
                for scene in SCENARIOS:
                    can = speed_dir / f"candidate_{scene.name}"
                    can.mkdir()
                    x = np.zeros((2,7),dtype=float)
                    x[:,0] = (60 if state["wrong_speed"] else 40)/3.6
                    np.savez_compressed(can/"candidate.npz",state_si=x)
                    (can/"candidate_report.json").write_text(
                        json.dumps({
                            "capture_sim_sha256":"s40",
                            "capture_parameter_sha256":"p40",
                        }),encoding="utf-8"
                    )
                    ep = workspace / "episodes" / scene.name
                    ep.mkdir(parents=True)
                    (ep / "experimental_episode.json").write_text(
                        json.dumps({"candidate_path":str(can.resolve())}),
                        encoding="utf-8"
                    )
                    rows.append({
                        "name":scene.name,
                        "episode_path":str(ep.resolve()),
                    })
                (workspace/"registry.json").write_text(
                    json.dumps({
                        "episode_count":len(rows),
                        "episodes":rows,
                    }),encoding="utf-8"
                )
                (speed_dir/"steer_calibration.json").write_text(
                    json.dumps({
                        "positive":{"initial_speed_kmh":40.0},
                        "negative":{"initial_speed_kmh":40.0},
                    }),encoding="utf-8"
                )
                (speed_dir/"coverage_report.json").write_text(
                    '{"passed":true}',encoding="utf-8"
                )

            with patch.object(m,"_audit_and_probe",
                              return_value=({40:fake_sim},audit,{40:40.0})):
                with patch("wm_recovery.scenario_runner.capture_all",
                           side_effect=fake_capture_all):
                    m.capture_speeds(temp,target,(40,))
                    self.assertTrue(
                        (target/"40kmh"/"speed_capture_manifest.json")
                        .is_file()
                    )
                    m.capture_speeds(temp,target,(40,),resume=True)
                    self.assertEqual(state["calls"],1)

            # 第二个批次故意制造首帧速度不符，应被拒绝。
            state["wrong_speed"] = True
            with patch.object(m,"_audit_and_probe",
                              return_value=({40:fake_sim},audit,{40:40.0})):
                with patch("wm_recovery.scenario_runner.capture_all",
                           side_effect=fake_capture_all):
                    with self.assertRaisesRegex(ValueError,"实际首帧"):
                        m.capture_speeds(
                            temp, Path(temp)/"batch_wrong", (40,)
                        )

if __name__ == "__main__":
    unittest.main()

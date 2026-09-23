"""CarSim path-tracking wrapper shared by training and deployed controllers."""

from typing import Tuple

import numpy as np

from carsim_env import CarSimEnv
from frenet_path import FrenetPath


OBS_DIM = 7
MAX_STEER_DEG = 90.0
DEFAULT_TARGET_SPEED_KMH = 40.0
DEFAULT_PATH_FILE = (
	r"E:\modelcodeE\carsim2022\CarSim2022.1_Data"
	r"\Roads\Center_XY\PathXY_6d2b4643-5dda-4b55-b5aa-1eaec807edf2.par"
)
CONTROL_DT_S = 0.05
MAX_LONGITUDINAL_RATE_PER_S = 2.0
MAX_STEER_RATE_DEG_PER_S = 180.0
WARMUP_HOLD_STEPS = 10
WARMUP_LONGITUDINAL_LIMIT = 0.5
WARMUP_STEER_LIMIT = 0.5
PATH_END_TOLERANCE_M = 1.0
PATH_END_MAX_LATERAL_ERROR_M = 2.0


class CarSimGymWrapper:
	"""基于 CarSimEnv 的路径跟踪环境包装。

	动作接口：
	- raw_action ∈ [-1,1]^2：
	  * raw_action[0] = lon：纵向控制（油门/制动合并）。
	  * raw_action[1] = steer_raw：方向盘控制。

	映射到仿真环境：
	- 若 lon >= 0：throttle = lon ∈ [0,1], brake = 0；
	- 若 lon < 0：throttle = 0, brake = -lon * 10.0 ∈ [0,10]；
	- steer = steer_raw * MAX_STEER_DEG ∈ [-90, 90] 度。
	- 执行动作受纵向命令变化率和方向盘角速率限制。

	观测：
	- 不再包含行驶距离 s，避免策略“背题”；
	- 组成：
	  [lat_n, heading_n, yaw_rate_n, spd_n, tgt_spd_n, lon_prev, steer_prev]
	  其中：
	    * lat_n：横向误差归一化到 [-1,1]；
	    * heading_n：航向误差（±45度）归一化到 [-1,1]；
	    * yaw_rate_n：横摆角速度（±60度/秒）归一化到 [-1,1]；
	    * spd_n：车速 (0~160km/h) 归一化到 [-1,1]；
	    * tgt_spd_n：目标车速（默认 40km/h）归一化到 [-1,1]；
	    * lon_prev, steer_prev：上一帧执行的动作（仍然在 [-1,1]）。
	"""

	def __init__(
			self,
			sim_path: str,
			target_speed: float = DEFAULT_TARGET_SPEED_KMH,
			path_file: str = DEFAULT_PATH_FILE,
	):
		self._env = CarSimEnv(sim_path)
		self._path = FrenetPath(path_file)

		self.n_import = None
		self.last_dist = 0.0
		self.target_speed = float(target_speed)
		self.prev_action = np.zeros(2, dtype=np.float32)
		self.current_state = None

	def reset(self) -> np.ndarray:
		exports = self._env.reset()
		self.n_import = self._env.n_import

		state = self._extract_state(exports)
		self.current_state = state
		self.last_dist = state["station"]
		self.prev_action[:] = 0.0

		return self._process_state(state)

	def step(
			self,
			raw_action: np.ndarray,
	) -> Tuple[np.ndarray, float, bool, dict]:

		action = np.asarray(
			raw_action,
			dtype=np.float64,
		).ravel()

		if action.size < 2:
			action = np.pad(
				action,
				(0, 2 - action.size),
				"constant",
			)

		action = np.clip(action[:2], -1.0, 1.0)

		desired_lon = float(action[0])
		desired_steer_raw = float(action[1])
		lon, steer_raw = self._rate_limit_action(
			desired_lon,
			desired_steer_raw,
		)

		if lon >= 0.0:
			throttle = lon
			brake = 0.0
		else:
			throttle = 0.0
			brake = -lon * 10.0

		steer = steer_raw * MAX_STEER_DEG

		env_action = [
			throttle,
			brake,
			steer,
		]

		control_dt = CONTROL_DT_S
		t_step = self._env.t_step

		if t_step <= 0.0:
			t_step = 0.001

		inner_steps = max(
			1,
			int(round(control_dt / t_step)),
		)

		exports, _, done, info = self._env.control_step(
			env_action,
			inner_steps=inner_steps,
		)

		state = self._extract_state(exports)
		self.current_state = state

		reward = self._compute_reward(
			state,
			desired_lon,
			desired_steer_raw,
		)

		# 下一状态应包含本次实际执行的动作，避免滞后一个控制周期。
		self.prev_action = np.array(
			[lon, steer_raw],
			dtype=np.float32,
		)
		obs = self._process_state(state)

		lat = state["lateral_error"]

		if not isinstance(info, dict):
			info = {"solver_info": info}

		info["lateral_error"] = lat
		info["heading_error_deg"] = state["heading_error_deg"]
		info["station"] = state["station"]
		info["speed_kmh"] = state["speed"]
		info["x"] = state["x"]
		info["y"] = state["y"]
		info["yaw_deg"] = state["yaw_deg"]
		info["yaw_rate_deg_s"] = state["yaw_rate_deg_s"]
		info["path_length"] = self._path.path_length
		info["path_progress"] = float(np.clip(
			state["station"] / max(self._path.path_length, 1e-6),
			0.0,
			1.0,
		))
		info["desired_action"] = (desired_lon, desired_steer_raw)
		info["executed_action"] = (lon, steer_raw)
		info["applied_controls"] = (throttle, brake, steer)

		# 路径投影会把超过终点的位置钳制在最后一个点；因此必须先判断
		# “到达终点”，否则驶过终点的纵向距离可能被当成横向误差。
		if self._is_path_complete(state):
			completion_bonus = 50.0
			reward += completion_bonus
			done = True
			info["completed"] = True
			info["completion_reason"] = "path_end"
			info["completion_bonus"] = completion_bonus
		elif abs(lat) > 4.0:
			lane_penalty = 50.0
			reward -= lane_penalty
			done = True

			info["off_track"] = True
			info["lane_penalty"] = lane_penalty
		elif (
			done
			and self._env.t_stop > 0.0
			and self._env.t_current >= self._env.t_stop - self._env.t_step
		):
			completion_bonus = 50.0
			reward += completion_bonus
			info["completed"] = True
			info["completion_reason"] = "simulation_end"
			info["completion_bonus"] = completion_bonus

		return obs, reward, done, info

	def _is_path_complete(self, state: dict) -> bool:
		"""车辆在路径末端附近且仍位于道路范围内时判定为成功。"""
		return (
			state["station"] >= self._path.path_length - PATH_END_TOLERANCE_M
			and abs(state["lateral_error"]) <= PATH_END_MAX_LATERAL_ERROR_M
		)

	def _rate_limit_action(self, desired_lon: float, desired_steer_raw: float):
		"""根据上一次执行动作限制本周期的控制变化率。"""
		previous_lon = float(self.prev_action[0])
		previous_steer_raw = float(self.prev_action[1])
		max_lon_delta = MAX_LONGITUDINAL_RATE_PER_S * CONTROL_DT_S
		max_steer_raw_delta = (
			MAX_STEER_RATE_DEG_PER_S * CONTROL_DT_S / MAX_STEER_DEG
		)

		lon = float(np.clip(
			desired_lon,
			previous_lon - max_lon_delta,
			previous_lon + max_lon_delta,
		))
		steer_raw = float(np.clip(
			desired_steer_raw,
			previous_steer_raw - max_steer_raw_delta,
			previous_steer_raw + max_steer_raw_delta,
		))
		return lon, steer_raw

	def close(self):
		try:
			self._env.close()
		except Exception:
			pass

	def _extract_state(self, exports):
		exports = list(exports)

		if len(exports) < 8:
			raise RuntimeError(
				f"Expected 8 CarSim exports, got {len(exports)}"
			)

		x = float(exports[0])
		y = float(exports[1])
		yaw_deg = float(exports[2])
		speed = float(exports[6])

		frenet = self._path.project(
			x,
			y,
			yaw_deg,
		)

		return {
			"x": x,
			"y": y,
			"yaw_deg": yaw_deg,
			"yaw_rate_deg_s": float(exports[3]),
			"lateral_error": frenet["lateral_error"],
			"heading_error_deg": frenet["heading_error_deg"],
			"station": frenet["station"],
			"speed": speed,
		}

	def _process_state(self, state):
		lat = state["lateral_error"]
		heading_error_deg = state["heading_error_deg"]
		yaw_rate_deg_s = state["yaw_rate_deg_s"]
		speed = state["speed"]

		lat_n = np.clip(
			lat / 5.0,
			-1.0,
			1.0,
		)

		heading_n = np.clip(
			heading_error_deg / 45.0,
			-1.0,
			1.0,
		)

		yaw_rate_n = np.clip(
			yaw_rate_deg_s / 60.0,
			-1.0,
			1.0,
		)

		speed_n = np.clip(
			speed / 160.0,
			0.0,
			1.0,
		)
		speed_n = 2.0 * speed_n - 1.0

		target_speed_n = np.clip(
			self.target_speed / 160.0,
			0.0,
			1.0,
		)
		target_speed_n = 2.0 * target_speed_n - 1.0

		lon_prev, steer_prev = self.prev_action

		return np.array(
			[
				lat_n,
				heading_n,
				yaw_rate_n,
				speed_n,
				target_speed_n,
				lon_prev,
				steer_prev,
			],
			dtype=np.float32,
		)

	def _compute_reward(
			self,
			state,
			lon: float,
			steer_raw: float,
	) -> float:

		lat = state["lateral_error"]
		heading_error_deg = state["heading_error_deg"]
		speed = state["speed"]
		distance = state["station"]

		delta_dist = distance - self.last_dist
		self.last_dist = distance

		expected_step_dist = max(
			self.target_speed / 3.6 * CONTROL_DT_S,
			0.1,
		)
		progress_reward = np.clip(
			delta_dist / expected_step_dist,
			-1.0,
			1.5,
		)
		lat_pen = (lat / 4.0) ** 2
		heading_pen = (heading_error_deg / 45.0) ** 2

		speed_error = (
							  speed - self.target_speed
					  ) / max(self.target_speed, 1.0)

		speed_pen = speed_error ** 2

		low_speed_pen = 0.0
		if speed < 3.0:
			low_speed_pen = (
									(3.0 - speed) / 3.0
							) ** 2

		action_pen = lon ** 2 + steer_raw ** 2

		delta_steer = (
				steer_raw
				- self.prev_action[1]
		)
		delta_steer_pen = delta_steer ** 2

		no_move_pen = 0.0
		if delta_dist < 0.02:
			no_move_pen = 0.2

		reward = (
			progress_reward
			- 0.5 * lat_pen
			- 0.2 * heading_pen
			- 0.3 * speed_pen
			- 0.3 * low_speed_pen
			- 0.05 * action_pen
			- 0.1 * delta_steer_pen
			- no_move_pen
		)

		return float(reward)

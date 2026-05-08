#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <msgpack.hpp>

namespace fr3 {

constexpr int kStatePubPort = 5555;
constexpr int kActionPullPort = 5556;

constexpr double kPolicyHz = 15.0;
constexpr double kDefaultPolicyDt = 1.0 / kPolicyHz;
constexpr double kActionChunkStaleSec = 0.5;
constexpr std::size_t kJointDof = 7;
constexpr std::size_t kActionDimJointVelocity = 7;
constexpr std::size_t kActionDimWithGripper = 8;
constexpr std::size_t kDefaultActionHorizon = 16;

constexpr float kMaxJointVel = 1.5F;
constexpr float kGripperCloseThreshold = 0.5F;
constexpr double kGripperMaxWidth = 0.08;
constexpr double kGripperGraspWidth = 0.0;
constexpr double kGripperGraspSpeed = 0.1;
constexpr double kGripperGraspForce = 20.0;

constexpr std::array<double, kJointDof> kJointLimitsLow = {
    -2.85, -1.76, -2.85, -3.04, -2.85, 0.40, -2.85};
constexpr std::array<double, kJointDof> kJointLimitsHigh = {
    2.85, 1.76, 2.85, -0.15, 2.85, 3.65, 2.85};

struct RobotStateMsg {
  std::array<float, 3> eef_pos{};
  std::array<float, 4> eef_quat_xyzw{{0.0F, 0.0F, 0.0F, 1.0F}};
  float gripper_width = static_cast<float>(kGripperMaxWidth);
  std::array<float, kJointDof> joint_position{};
  double capture_ts = 0.0;
};

struct ActionChunk {
  std::vector<float> actions;
  std::size_t rows = 0;
  std::size_t cols = 0;
  double capture_ts = 0.0;
  double policy_dt = kDefaultPolicyDt;

  float action(std::size_t row, std::size_t col) const {
    return actions.at(row * cols + col);
  }

  bool has_gripper_command() const {
    return cols >= kActionDimWithGripper;
  }
};

inline double wall_time_seconds() {
  using clock = std::chrono::system_clock;
  const auto now = clock::now().time_since_epoch();
  return std::chrono::duration<double>(now).count();
}

inline double steady_time_seconds() {
  using clock = std::chrono::steady_clock;
  const auto now = clock::now().time_since_epoch();
  return std::chrono::duration<double>(now).count();
}

inline float clamp_float(float value, float low, float high) {
  return std::min(std::max(value, low), high);
}

inline std::array<double, kJointDof> clamp_joint_position(std::array<double, kJointDof> q) {
  for (std::size_t i = 0; i < q.size(); ++i) {
    q[i] = std::min(std::max(q[i], kJointLimitsLow[i]), kJointLimitsHigh[i]);
  }
  return q;
}

inline std::array<float, kJointDof> clamp_joint_velocity_row(const float* action) {
  std::array<float, kJointDof> out{};
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = clamp_float(action[i], -kMaxJointVel, kMaxJointVel);
  }
  return out;
}

inline std::array<float, kActionDimWithGripper> clamp_action_droid(const float* action) {
  std::array<float, kActionDimWithGripper> out{};
  const auto joint_velocity = clamp_joint_velocity_row(action);
  for (std::size_t i = 0; i < joint_velocity.size(); ++i) {
    out[i] = joint_velocity[i];
  }
  out[7] = clamp_float(action[7], 0.0F, 1.0F);
  return out;
}

inline const msgpack::object* map_find(const msgpack::object& obj, const std::string& key) {
  if (obj.type != msgpack::type::MAP) {
    throw std::runtime_error("msgpack object is not a map");
  }
  for (std::uint32_t i = 0; i < obj.via.map.size; ++i) {
    const auto& item = obj.via.map.ptr[i];
    if (item.key.type == msgpack::type::STR && item.key.as<std::string>() == key) {
      return &item.val;
    }
  }
  return nullptr;
}

inline const msgpack::object& require_key(const msgpack::object& obj, const std::string& key) {
  const msgpack::object* value = map_find(obj, key);
  if (value == nullptr) {
    throw std::runtime_error("missing msgpack key: " + key);
  }
  return *value;
}

inline std::pair<const char*, std::size_t> require_bin(const msgpack::object& obj,
                                                       const std::string& key) {
  const auto& value = require_key(obj, key);
  if (value.type == msgpack::type::BIN) {
    return {value.via.bin.ptr, value.via.bin.size};
  }
  if (value.type == msgpack::type::STR) {
    return {value.via.str.ptr, value.via.str.size};
  }
  throw std::runtime_error("msgpack key is not bin/str: " + key);
}

inline ActionChunk unpack_action_chunk(const char* data, std::size_t size) {
  msgpack::object_handle handle = msgpack::unpack(data, size);
  const msgpack::object& obj = handle.get();

  const auto& shape_obj = require_key(obj, "actions_shape");
  if (shape_obj.type != msgpack::type::ARRAY || shape_obj.via.array.size != 2) {
    throw std::runtime_error("actions_shape must be an array of length 2");
  }

  ActionChunk chunk;
  chunk.rows = shape_obj.via.array.ptr[0].as<std::size_t>();
  chunk.cols = shape_obj.via.array.ptr[1].as<std::size_t>();
  if (chunk.rows == 0 ||
      (chunk.cols != kActionDimJointVelocity && chunk.cols != kActionDimWithGripper)) {
    throw std::runtime_error("expected actions shape (N, 7) or (N, 8)");
  }

  const auto [actions_ptr, actions_bytes] = require_bin(obj, "actions");
  const std::size_t expected_bytes = chunk.rows * chunk.cols * sizeof(float);
  if (actions_bytes != expected_bytes) {
    throw std::runtime_error("actions byte size does not match actions_shape");
  }
  chunk.actions.resize(chunk.rows * chunk.cols);
  std::memcpy(chunk.actions.data(), actions_ptr, actions_bytes);

  chunk.capture_ts = require_key(obj, "capture_ts").as<double>();
  const msgpack::object* policy_dt = map_find(obj, "policy_dt");
  if (policy_dt != nullptr) {
    chunk.policy_dt = policy_dt->as<double>();
  }
  return chunk;
}

template <typename T>
inline void pack_bin(msgpack::packer<msgpack::sbuffer>& packer, const T* data, std::size_t count) {
  const auto bytes = static_cast<std::uint32_t>(sizeof(T) * count);
  packer.pack_bin(bytes);
  packer.pack_bin_body(reinterpret_cast<const char*>(data), bytes);
}

inline std::string pack_robot_state(const RobotStateMsg& state) {
  msgpack::sbuffer buffer;
  msgpack::packer<msgpack::sbuffer> packer(buffer);

  packer.pack_map(5);
  packer.pack(std::string("eef_pos"));
  pack_bin(packer, state.eef_pos.data(), state.eef_pos.size());
  packer.pack(std::string("eef_quat_xyzw"));
  pack_bin(packer, state.eef_quat_xyzw.data(), state.eef_quat_xyzw.size());
  packer.pack(std::string("gripper_width"));
  packer.pack(state.gripper_width);
  packer.pack(std::string("joint_position"));
  pack_bin(packer, state.joint_position.data(), state.joint_position.size());
  packer.pack(std::string("capture_ts"));
  packer.pack(state.capture_ts);

  return std::string(buffer.data(), buffer.size());
}

inline std::array<float, 4> rotation_to_quat_xyzw(const std::array<double, 16>& transform) {
  const double r00 = transform[0];
  const double r01 = transform[4];
  const double r02 = transform[8];
  const double r10 = transform[1];
  const double r11 = transform[5];
  const double r12 = transform[9];
  const double r20 = transform[2];
  const double r21 = transform[6];
  const double r22 = transform[10];

  std::array<double, 4> q{};
  const double trace = r00 + r11 + r22;
  if (trace > 0.0) {
    const double s = std::sqrt(trace + 1.0) * 2.0;
    q = {(r21 - r12) / s, (r02 - r20) / s, (r10 - r01) / s, 0.25 * s};
  } else if (r00 > r11 && r00 > r22) {
    const double s = std::sqrt(1.0 + r00 - r11 - r22) * 2.0;
    q = {0.25 * s, (r01 + r10) / s, (r02 + r20) / s, (r21 - r12) / s};
  } else if (r11 > r22) {
    const double s = std::sqrt(1.0 + r11 - r00 - r22) * 2.0;
    q = {(r01 + r10) / s, 0.25 * s, (r12 + r21) / s, (r02 - r20) / s};
  } else {
    const double s = std::sqrt(1.0 + r22 - r00 - r11) * 2.0;
    q = {(r02 + r20) / s, (r12 + r21) / s, 0.25 * s, (r10 - r01) / s};
  }

  const double norm = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (norm < 1e-12) {
    return {0.0F, 0.0F, 0.0F, 1.0F};
  }
  return {static_cast<float>(q[0] / norm), static_cast<float>(q[1] / norm),
          static_cast<float>(q[2] / norm), static_cast<float>(q[3] / norm)};
}

inline std::array<double, kJointDof> integrate_joint_velocity(
    const ActionChunk& chunk,
    const std::array<double, kJointDof>& q_start) {
  std::array<double, kJointDof> q = q_start;
  for (std::size_t row = 0; row < chunk.rows; ++row) {
    const auto velocity = clamp_joint_velocity_row(&chunk.actions[row * chunk.cols]);
    for (std::size_t i = 0; i < q.size(); ++i) {
      q[i] += static_cast<double>(velocity[i]) * chunk.policy_dt;
    }
    q = clamp_joint_position(q);
  }
  return q;
}

inline std::vector<std::array<double, kJointDof>> integrate_joint_velocity_waypoints(
    const ActionChunk& chunk,
    const std::array<double, kJointDof>& q_start) {
  std::array<double, kJointDof> q = q_start;
  std::vector<std::array<double, kJointDof>> waypoints;
  waypoints.reserve(chunk.rows);
  for (std::size_t row = 0; row < chunk.rows; ++row) {
    const auto velocity = clamp_joint_velocity_row(&chunk.actions[row * chunk.cols]);
    for (std::size_t i = 0; i < q.size(); ++i) {
      q[i] += static_cast<double>(velocity[i]) * chunk.policy_dt;
    }
    q = clamp_joint_position(q);
    waypoints.push_back(q);
  }
  return waypoints;
}

}  // namespace fr3

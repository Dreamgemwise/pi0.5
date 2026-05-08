#include "common.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <csignal>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <franka/control_types.h>
#include <franka/duration.h>
#include <franka/exception.h>
#include <franka/gripper.h>
#include <franka/rate_limiting.h>
#include <franka/robot.h>
#include <franka/robot_state.h>
#include <zmq.hpp>

namespace {

std::atomic<bool> g_stop{false};

void handle_signal(int) {
  g_stop.store(true);
}

struct Args {
  std::string robot_ip;
  std::string bind = "0.0.0.0";
  bool no_gripper = false;
  bool ignore_realtime_check = false;
  double dynamics_factor = 0.05;
  double ema_alpha = 0.1;
  franka::ControllerMode controller_mode = franka::ControllerMode::kCartesianImpedance;
};

void usage(const char* argv0) {
  std::cerr << "Usage: " << argv0
            << " --robot-ip 172.16.0.2 [--bind 0.0.0.0] [--no-gripper]\n"
            << "       [--ignore-realtime-check] [--controller-mode cartesian|joint]\n"
            << "       [--dynamics-factor 0.05] [--ema-alpha 0.35]\n";
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto require_value = [&](const std::string& option) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + option);
      }
      return argv[++i];
    };

    if (key == "--robot-ip") {
      args.robot_ip = require_value(key);
    } else if (key == "--bind") {
      args.bind = require_value(key);
    } else if (key == "--no-gripper") {
      args.no_gripper = true;
    } else if (key == "--ignore-realtime-check") {
      args.ignore_realtime_check = true;
    } else if (key == "--dynamics-factor") {
      args.dynamics_factor = std::stod(require_value(key));
    } else if (key == "--ema-alpha") {
      args.ema_alpha = std::stod(require_value(key));
    } else if (key == "--controller-mode") {
      const std::string mode = require_value(key);
      if (mode == "joint") {
        args.controller_mode = franka::ControllerMode::kJointImpedance;
      } else if (mode == "cartesian") {
        args.controller_mode = franka::ControllerMode::kCartesianImpedance;
      } else {
        throw std::runtime_error("unknown controller mode: " + mode);
      }
    } else if (key == "--help" || key == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + key);
    }
  }

  if (args.robot_ip.empty()) {
    throw std::runtime_error("--robot-ip is required");
  }
  if (args.dynamics_factor <= 0.0 || args.dynamics_factor > 1.0) {
    throw std::runtime_error("--dynamics-factor must be in (0, 1]");
  }
  if (args.ema_alpha <= 0.0 || args.ema_alpha > 1.0) {
    throw std::runtime_error("--ema-alpha must be in (0, 1]");
  }
  return args;
}

void set_collision_behavior(franka::Robot& robot) {
  const std::array<double, 7> lower_torque{{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}};
  const std::array<double, 7> upper_torque{{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}};
  const std::array<double, 6> lower_force{{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}};
  const std::array<double, 6> upper_force{{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}};
  robot.setCollisionBehavior(lower_torque, upper_torque, lower_force, upper_force);
}

struct ActionCommand {
  std::vector<std::array<float, fr3::kJointDof>> joint_velocities;
  bool has_gripper_command = false;
  float gripper_command = 0.0F;
  double policy_dt = fr3::kDefaultPolicyDt;
  std::uint64_t seq = 0;
};

class RobotServer {
 public:
  explicit RobotServer(const Args& args)
      : args_(args),
        ctx_(1),
        state_pub_(ctx_, zmq::socket_type::pub),
        action_pull_(ctx_, zmq::socket_type::pull),
        robot_(args.robot_ip,
               args.ignore_realtime_check ? franka::RealtimeConfig::kIgnore
                                          : franka::RealtimeConfig::kEnforce) {
    std::cout << "Connecting to Franka " << args.robot_ip << " with C++ libfranka\n";
    set_collision_behavior(robot_);

    state_pub_.bind("tcp://" + args.bind + ":" + std::to_string(fr3::kStatePubPort));
    action_pull_.bind("tcp://" + args.bind + ":" + std::to_string(fr3::kActionPullPort));
    action_pull_.set(zmq::sockopt::rcvtimeo, 50);

    if (!args.no_gripper) {
      try {
        gripper_ = std::make_unique<franka::Gripper>(args.robot_ip);
      } catch (const std::exception& exc) {
        std::cerr << "Gripper unavailable: " << exc.what() << "\n";
      }
    }

    std::cout << "ZMQ bound: PUB state tcp://" << args.bind << ":" << fr3::kStatePubPort
              << ", PULL action tcp://" << args.bind << ":" << fr3::kActionPullPort << "\n";
    pending_action_.joint_velocities.reserve(fr3::kDefaultActionHorizon);
    active_action_.joint_velocities.reserve(fr3::kDefaultActionHorizon);
    active_waypoints_.reserve(fr3::kDefaultActionHorizon);
  }

  void run() {
    std::thread pub_thread(&RobotServer::state_pub_loop, this);
    std::thread action_thread(&RobotServer::action_pull_loop, this);
    std::thread gripper_thread;
    if (gripper_) {
      gripper_thread = std::thread(&RobotServer::gripper_state_loop, this);
    }

    try {
      robot_.control(
          [this](const franka::RobotState& state, franka::Duration period) -> franka::JointPositions {
            update_cached_state(state);
            if (!have_hold_q_) {
              hold_q_ = state.q;
              have_hold_q_ = true;
            }
            adopt_pending_action(state);
            const auto target = target_from_active_action(period, hold_q_);
            hold_q_ = rate_limit_joint_positions(target, hold_q_, state);
            franka::JointPositions command(hold_q_);
            if (g_stop.load()) {
              return franka::MotionFinished(command);
            }
            return command;
          },
          args_.controller_mode);
    } catch (const franka::Exception& exc) {
      if (!g_stop.load()) {
        std::cerr << "libfranka control exception: " << exc.what() << "\n";
      }
    }

    g_stop.store(true);
    if (pub_thread.joinable()) {
      pub_thread.join();
    }
    if (action_thread.joinable()) {
      action_thread.join();
    }
    if (gripper_thread.joinable()) {
      gripper_thread.join();
    }
  }

 private:
  float cached_gripper_width() const {
    std::lock_guard<std::mutex> lock(gripper_width_mutex_);
    return static_cast<float>(std::min(std::max(gripper_width_, 0.0), fr3::kGripperMaxWidth));
  }

  void update_cached_state(const franka::RobotState& robot_state) {
    fr3::RobotStateMsg msg;
    msg.capture_ts = fr3::wall_time_seconds();
    msg.eef_pos = {static_cast<float>(robot_state.O_T_EE[12]),
                   static_cast<float>(robot_state.O_T_EE[13]),
                   static_cast<float>(robot_state.O_T_EE[14])};
    msg.eef_quat_xyzw = fr3::rotation_to_quat_xyzw(robot_state.O_T_EE);
    msg.gripper_width = cached_gripper_width();
    for (std::size_t i = 0; i < fr3::kJointDof; ++i) {
      msg.joint_position[i] = static_cast<float>(robot_state.q[i]);
    }

    std::lock_guard<std::mutex> lock(state_mutex_);
    latest_state_ = msg;
    have_state_ = true;
  }

  void state_pub_loop() {
    constexpr auto period = std::chrono::milliseconds(10);
    while (!g_stop.load()) {
      const auto t0 = std::chrono::steady_clock::now();
      fr3::RobotStateMsg state;
      bool have_state = false;
      {
        std::lock_guard<std::mutex> lock(state_mutex_);
        state = latest_state_;
        have_state = have_state_;
      }

      if (have_state) {
        const std::string payload = fr3::pack_robot_state(state);
        zmq::message_t message(payload.size());
        std::memcpy(message.data(), payload.data(), payload.size());
        state_pub_.send(message, zmq::send_flags::none);
      }

      const auto elapsed = std::chrono::steady_clock::now() - t0;
      if (elapsed < period) {
        std::this_thread::sleep_for(period - elapsed);
      }
    }
  }

  void action_pull_loop() {
    while (!g_stop.load()) {
      zmq::message_t message;
      const auto result = action_pull_.recv(message, zmq::recv_flags::none);
      if (!result) {
        continue;
      }

      try {
        const auto chunk = fr3::unpack_action_chunk(static_cast<const char*>(message.data()),
                                                    message.size());
        const double age = fr3::wall_time_seconds() - chunk.capture_ts;
        if (age > fr3::kActionChunkStaleSec) {
          std::cerr << "Drop stale chunk: age=" << age << "s\n";
          continue;
        }
        if (chunk.policy_dt <= 0.0 || chunk.policy_dt > 1.0) {
          std::cerr << "Drop chunk with invalid policy_dt=" << chunk.policy_dt << "s\n";
          continue;
        }

        ActionCommand command;
        command.joint_velocities.reserve(chunk.rows);
        command.policy_dt = chunk.policy_dt;
        command.has_gripper_command = chunk.has_gripper_command();
        if (command.has_gripper_command) {
          command.gripper_command =
              fr3::clamp_float(chunk.action(chunk.rows - 1, 7), 0.0F, 1.0F);
          request_gripper(command.gripper_command);
        }

        double max_velocity_norm = 0.0;
        for (std::size_t row = 0; row < chunk.rows; ++row) {
          const auto velocity = fr3::clamp_joint_velocity_row(&chunk.actions[row * chunk.cols]);
          double velocity_norm = 0.0;
          for (double value : velocity) {
            velocity_norm += value * value;
          }
          max_velocity_norm = std::max(max_velocity_norm, std::sqrt(velocity_norm));
          command.joint_velocities.push_back(velocity);
        }

        {
          std::lock_guard<std::mutex> lock(action_mutex_);
          command.seq = ++pending_action_seq_;
          pending_action_ = command;
          have_pending_action_ = true;
        }

        std::cout << "chunk accepted: N=" << chunk.rows << ", age=" << age * 1000.0
                  << "ms, policy_dt=" << chunk.policy_dt
                  << "s, max|dq/dt|=" << max_velocity_norm << "rad/s\n";
      } catch (const std::exception& exc) {
        std::cerr << "Bad action chunk, drop: " << exc.what() << "\n";
      }
    }
  }

  void adopt_pending_action(const franka::RobotState& state) {
    ActionCommand incoming;
    bool have_incoming = false;
    {
      std::lock_guard<std::mutex> lock(action_mutex_);
      if (have_pending_action_ && pending_action_.seq != active_action_seq_) {
        incoming = pending_action_;
        have_incoming = true;
      }
    }

    if (!have_incoming || incoming.joint_velocities.empty()) {
      return;
    }

    active_action_ = incoming;
    active_action_seq_ = incoming.seq;
    active_step_index_ = 0;
    active_step_elapsed_ = 0.0;
    rebuild_active_waypoints(state.q, active_action_);
    have_active_action_ = !active_waypoints_.empty();
  }

  std::array<double, fr3::kJointDof> target_from_active_action(
      franka::Duration period,
      const std::array<double, fr3::kJointDof>& fallback) {
    if (!have_active_action_ || active_waypoints_.empty()) {
      return fallback;
    }

    active_step_elapsed_ += period.toSec();
    while (active_step_elapsed_ >= active_action_.policy_dt) {
      active_step_elapsed_ -= active_action_.policy_dt;
      if (active_step_index_ + 1 >= active_waypoints_.size()) {
        have_active_action_ = false;
        active_action_.joint_velocities.clear();
        active_waypoints_.clear();
        return fallback;
      }
      ++active_step_index_;
    }

    return active_waypoints_[active_step_index_];
  }

  void rebuild_active_waypoints(
      const std::array<double, fr3::kJointDof>& q_start,
      const ActionCommand& command) {
    std::array<double, fr3::kJointDof> q = q_start;
    active_waypoints_.clear();
    if (active_waypoints_.capacity() < command.joint_velocities.size()) {
      active_waypoints_.reserve(command.joint_velocities.size());
    }
    for (const auto& raw_velocity : command.joint_velocities) {
      for (std::size_t i = 0; i < ema_velocity_.size(); ++i) {
        ema_velocity_[i] = args_.ema_alpha * static_cast<double>(raw_velocity[i]) +
                           (1.0 - args_.ema_alpha) * ema_velocity_[i];
        q[i] += ema_velocity_[i] * command.policy_dt;
      }
      q = fr3::clamp_joint_position(q);
      active_waypoints_.push_back(q);
    }
  }

  std::array<double, fr3::kJointDof> rate_limit_joint_positions(
      const std::array<double, fr3::kJointDof>& target,
      const std::array<double, fr3::kJointDof>& last_command,
      const franka::RobotState& state) {
    auto upper_velocity = robot_.getUpperJointVelocityLimits(last_command);
    auto lower_velocity = robot_.getLowerJointVelocityLimits(last_command);
    auto max_acceleration = franka::kMaxJointAcceleration;
    auto max_jerk = franka::kMaxJointJerk;
    for (std::size_t i = 0; i < fr3::kJointDof; ++i) {
      upper_velocity[i] *= args_.dynamics_factor;
      lower_velocity[i] *= args_.dynamics_factor;
      max_acceleration[i] *= args_.dynamics_factor;
      max_jerk[i] *= args_.dynamics_factor;
    }
    return franka::limitRate(upper_velocity, lower_velocity, max_acceleration, max_jerk,
                             target, last_command, state.dq_d, state.ddq_d);
  }

  void request_gripper(float command) {
    if (!gripper_) {
      return;
    }
    const bool want_close = command > fr3::kGripperCloseThreshold;
    {
      std::lock_guard<std::mutex> lock(gripper_cmd_mutex_);
      if (gripper_pending_ || want_close == gripper_closed_) {
        return;
      }
      gripper_pending_ = true;
      gripper_closed_ = want_close;
    }

    std::thread([this, want_close]() {
      try {
        std::lock_guard<std::mutex> lock(gripper_io_mutex_);
        if (want_close) {
          gripper_->grasp(fr3::kGripperGraspWidth, fr3::kGripperGraspSpeed,
                          fr3::kGripperGraspForce, 0.08, 0.08);
        } else {
          gripper_->move(fr3::kGripperMaxWidth, fr3::kGripperGraspSpeed);
        }
      } catch (const std::exception& exc) {
        std::cerr << "Gripper command failed: " << exc.what() << "\n";
      }
      std::lock_guard<std::mutex> lock(gripper_cmd_mutex_);
      gripper_pending_ = false;
    }).detach();
  }

  void gripper_state_loop() {
    constexpr auto period = std::chrono::milliseconds(50);
    while (!g_stop.load()) {
      const auto t0 = std::chrono::steady_clock::now();
      try {
        std::lock_guard<std::mutex> io_lock(gripper_io_mutex_);
        const auto state = gripper_->readOnce();
        std::lock_guard<std::mutex> width_lock(gripper_width_mutex_);
        gripper_width_ = state.width;
      } catch (const std::exception&) {
      }

      const auto elapsed = std::chrono::steady_clock::now() - t0;
      if (elapsed < period) {
        std::this_thread::sleep_for(period - elapsed);
      }
    }
  }

  Args args_;
  zmq::context_t ctx_;
  zmq::socket_t state_pub_;
  zmq::socket_t action_pull_;
  franka::Robot robot_;
  std::unique_ptr<franka::Gripper> gripper_;

  mutable std::mutex state_mutex_;
  fr3::RobotStateMsg latest_state_;
  bool have_state_ = false;

  std::mutex action_mutex_;
  ActionCommand pending_action_;
  std::uint64_t pending_action_seq_ = 0;
  bool have_pending_action_ = false;

  ActionCommand active_action_;
  std::uint64_t active_action_seq_ = 0;
  bool have_active_action_ = false;
  std::size_t active_step_index_ = 0;
  double active_step_elapsed_ = 0.0;
  std::vector<std::array<double, fr3::kJointDof>> active_waypoints_;
  std::array<double, fr3::kJointDof> ema_velocity_{};
  std::array<double, fr3::kJointDof> hold_q_{};
  bool have_hold_q_ = false;

  mutable std::mutex gripper_width_mutex_;
  double gripper_width_ = fr3::kGripperMaxWidth;
  std::mutex gripper_io_mutex_;
  std::mutex gripper_cmd_mutex_;
  bool gripper_pending_ = false;
  bool gripper_closed_ = false;
};

}  // namespace

int main(int argc, char** argv) {
  std::signal(SIGINT, handle_signal);
  std::signal(SIGTERM, handle_signal);

  try {
    const Args args = parse_args(argc, argv);
    RobotServer(args).run();
  } catch (const std::exception& exc) {
    std::cerr << "robot_server failed: " << exc.what() << "\n";
    usage(argv[0]);
    return 1;
  }
  return 0;
}

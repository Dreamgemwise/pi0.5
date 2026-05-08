#include "common.hpp"

#include <atomic>
#include <cstdlib>
#include <csignal>
#include <iostream>
#include <stdexcept>
#include <string>

#include <franka/control_types.h>
#include <franka/duration.h>
#include <franka/exception.h>
#include <franka/gripper.h>
#include <franka/robot.h>
#include <franka/robot_state.h>

namespace {

std::atomic<bool> g_stop{false};

void handle_signal(int) {
  g_stop.store(true);
}

constexpr std::array<double, 7> kHomeJoints{{0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785}};

struct Args {
  std::string robot_ip = "172.16.0.2";
  double time_to_go = 8.0;
  bool no_gripper = false;
  bool ignore_realtime_check = false;
  franka::ControllerMode controller_mode = franka::ControllerMode::kCartesianImpedance;
};

void usage(const char* argv0) {
  std::cerr << "Usage: " << argv0
            << " [--robot-ip 172.16.0.2] [--time-to-go 8.0]\n"
            << "       [--no-gripper] [--ignore-realtime-check]\n"
            << "       [--controller-mode cartesian|joint]\n";
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
    } else if (key == "--time-to-go") {
      args.time_to_go = std::stod(require_value(key));
    } else if (key == "--no-gripper") {
      args.no_gripper = true;
    } else if (key == "--ignore-realtime-check") {
      args.ignore_realtime_check = true;
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
  return args;
}

void set_collision_behavior(franka::Robot& robot) {
  const std::array<double, 7> lower_torque{{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}};
  const std::array<double, 7> upper_torque{{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}};
  const std::array<double, 6> lower_force{{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}};
  const std::array<double, 6> upper_force{{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}};
  robot.setCollisionBehavior(lower_torque, upper_torque, lower_force, upper_force);
}

double smoothstep5(double t) {
  t = std::min(std::max(t, 0.0), 1.0);
  return 10.0 * std::pow(t, 3) - 15.0 * std::pow(t, 4) + 6.0 * std::pow(t, 5);
}

void open_gripper(const Args& args) {
  if (args.no_gripper) {
    return;
  }
  try {
    std::cout << "Opening gripper\n";
    franka::Gripper gripper(args.robot_ip);
    gripper.move(fr3::kGripperMaxWidth, fr3::kGripperGraspSpeed);
  } catch (const std::exception& exc) {
    std::cerr << "Open gripper failed, continuing go_home: " << exc.what() << "\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::signal(SIGINT, handle_signal);
  std::signal(SIGTERM, handle_signal);

  try {
    const Args args = parse_args(argc, argv);
    std::cout << "Connecting to Franka " << args.robot_ip << " with C++ libfranka\n";
    franka::Robot robot(args.robot_ip, args.ignore_realtime_check ? franka::RealtimeConfig::kIgnore
                                                                  : franka::RealtimeConfig::kEnforce);
    set_collision_behavior(robot);
    open_gripper(args);

    bool initialized = false;
    std::array<double, 7> q_start{};
    double elapsed = 0.0;

    robot.control(
        [&](const franka::RobotState& state, franka::Duration period) -> franka::JointPositions {
          if (!initialized) {
            q_start = state.q_d;
            initialized = true;
            std::cout << "Moving to home over " << args.time_to_go << "s\n";
          }

          elapsed += period.toSec();
          const double tau = std::min(1.0, elapsed / args.time_to_go);
          const double blend = smoothstep5(tau);

          std::array<double, 7> q_cmd{};
          for (std::size_t i = 0; i < 7; ++i) {
            q_cmd[i] = q_start[i] + blend * (kHomeJoints[i] - q_start[i]);
          }

          franka::JointPositions command(q_cmd);
          if (g_stop.load() || tau >= 1.0) {
            std::cout << "Home motion finished\n";
            return franka::MotionFinished(command);
          }
          return command;
        },
        args.controller_mode);
  } catch (const std::exception& exc) {
    std::cerr << "go_home failed: " << exc.what() << "\n";
    usage(argv[0]);
    return 1;
  }

  return 0;
}

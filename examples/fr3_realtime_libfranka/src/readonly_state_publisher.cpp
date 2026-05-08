#include "common.hpp"

#include <atomic>
#include <cstdlib>
#include <csignal>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include <franka/exception.h>
#include <franka/gripper.h>
#include <franka/robot.h>
#include <franka/robot_state.h>
#include <zmq.hpp>

namespace {

std::atomic<bool> g_stop{false};

void handle_signal(int) {
  g_stop.store(true);
}

struct Args {
  std::string robot_ip = "172.16.0.2";
  std::string bind = "0.0.0.0";
  double hz = 100.0;
  bool no_gripper = false;
  bool ignore_realtime_check = false;
  double default_gripper_width = fr3::kGripperMaxWidth;
};

void usage(const char* argv0) {
  std::cerr << "Usage: " << argv0
            << " [--robot-ip 172.16.0.2] [--bind 0.0.0.0] [--hz 100]\n"
            << "       [--no-gripper] [--ignore-realtime-check]\n";
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
    } else if (key == "--hz") {
      args.hz = std::stod(require_value(key));
    } else if (key == "--no-gripper") {
      args.no_gripper = true;
    } else if (key == "--ignore-realtime-check") {
      args.ignore_realtime_check = true;
    } else if (key == "--default-gripper-width") {
      args.default_gripper_width = std::stod(require_value(key));
    } else if (key == "--help" || key == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + key);
    }
  }
  return args;
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

    std::unique_ptr<franka::Gripper> gripper;
    double gripper_width = args.default_gripper_width;
    if (!args.no_gripper) {
      try {
        gripper = std::make_unique<franka::Gripper>(args.robot_ip);
        gripper_width = gripper->readOnce().width;
      } catch (const std::exception& exc) {
        std::cerr << "Gripper unavailable: " << exc.what() << "\n";
      }
    }

    zmq::context_t ctx(1);
    zmq::socket_t pub(ctx, zmq::socket_type::pub);
    pub.bind("tcp://" + args.bind + ":" + std::to_string(fr3::kStatePubPort));
    std::cout << "Publishing read-only RobotState on tcp://" << args.bind << ":"
              << fr3::kStatePubPort << " at " << args.hz << "Hz\n";

    const auto period = std::chrono::duration<double>(1.0 / args.hz);
    while (!g_stop.load()) {
      const auto t0 = std::chrono::steady_clock::now();
      const auto state = robot.readOnce();
      if (gripper) {
        try {
          gripper_width = gripper->readOnce().width;
        } catch (const std::exception&) {
        }
      }

      fr3::RobotStateMsg msg;
      msg.capture_ts = fr3::wall_time_seconds();
      msg.eef_pos = {static_cast<float>(state.O_T_EE[12]), static_cast<float>(state.O_T_EE[13]),
                     static_cast<float>(state.O_T_EE[14])};
      msg.eef_quat_xyzw = fr3::rotation_to_quat_xyzw(state.O_T_EE);
      msg.gripper_width =
          static_cast<float>(std::min(std::max(gripper_width, 0.0), fr3::kGripperMaxWidth));
      for (std::size_t i = 0; i < 7; ++i) {
        msg.joint_position[i] = static_cast<float>(state.q[i]);
      }

      const std::string payload = fr3::pack_robot_state(msg);
      zmq::message_t message(payload.size());
      std::memcpy(message.data(), payload.data(), payload.size());
      pub.send(message, zmq::send_flags::none);

      const auto elapsed = std::chrono::steady_clock::now() - t0;
      if (elapsed < period) {
        std::this_thread::sleep_for(period - elapsed);
      }
    }
  } catch (const std::exception& exc) {
    std::cerr << "readonly_state_publisher failed: " << exc.what() << "\n";
    usage(argv[0]);
    return 1;
  }

  return 0;
}

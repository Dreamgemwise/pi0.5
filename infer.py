import os
os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false'
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".70"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from pathlib import Path

from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download
import numpy as np
import time

import torch
from torch.profiler import profile, record_function, ProfilerActivity,schedule
torch._dynamo.config.suppress_errors = True


# import jax
# import jax.profiler

model_name = "pi05_libero"

print(f'Config [{model_name}]....')
config = _config.get_config(model_name)
checkpoint_dir = str(Path("models") / "pi05_libero_pytorch")
print(f'Load {model_name} done.')

def _random_observation_droid() -> dict:
    return {
        # "observation/image": np.random.rand(224, 224, 3).astype(np.float32),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        # "observation/wrist_image": np.random.rand(224, 224, 3).astype(np.float32),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.rand(8).astype(np.float32),
        # "prompt": "do something",
        "prompt": "",
    }

print('Generating example observation...')
example = _random_observation_droid()

# example["observation/image"] = "examples/assets/pic1.jfif"
# example["observation/wrist_image"] = "examples/assets/pic2.jfif"

print('Creating trained policy....')
policy = policy_config.create_trained_policy(config, checkpoint_dir)

latency_list=[]

my_dict=policy.infer(example)
action_chunk = my_dict["actions"]      # 预热模型避免造成统计偏差
latency=my_dict["policy_timing"]["infer_ms"]
latency_list.append(latency)
print(f'Warm-Up done, cost time {latency:.3f} ms')

print('-' * 50)

inference_count = 10
total_inference_time = 0.0

print('Inference...')
for i in range(inference_count):
    print('-' * 50)
    print(f"Ready to {i+1}/{inference_count} inference...")
    my_dict=policy.infer(example)
    torch.cuda.synchronize()
    action_chunk = my_dict["actions"]
    latency=my_dict["policy_timing"]["infer_ms"]
    timestamp=my_dict["policy_timing"]["timestamp_unix"]
    print(f'Inference done, cost time {latency:.3f} ms')
    print(action_chunk)
    total_inference_time += latency
    latency_list.append(latency)
    print(action_chunk.shape)
# print("jax.profiler.stop_trace ----done")
# jax.profiler.stop_trace()

print(f'Total inference done, average cost time: {(total_inference_time / inference_count)} ms')

print(latency_list)




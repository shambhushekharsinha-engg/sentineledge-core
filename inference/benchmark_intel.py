import os
import time
import numpy as np
import openvino as ov

def benchmark(ir_path="inference/ir_model/vla_policy.xml", device_name="AUTO", num_iterations=100):
    """
    Benchmarks the OpenVINO VLA model on Intel hardware.
    Device can be CPU, GPU (iGPU), or NPU.
    """
    if not os.path.exists(ir_path):
        print(f"Error: {ir_path} not found. Please run export_openvino.py first.")
        return

    core = ov.Core()
    print(f"Available devices: {core.available_devices}")
    
    print(f"Loading model to {device_name}...")
    model = core.read_model(ir_path)
    compiled_model = core.compile_model(model, device_name)
    
    # Setup inputs
    input_tensor_img = np.random.randn(1, 3, 480, 640).astype(np.float32)
    input_tensor_lang = np.random.randn(1, 768).astype(np.float32)
    
    print(f"Warming up for 10 iterations...")
    for _ in range(10):
        compiled_model([input_tensor_img, input_tensor_lang])
        
    print(f"Benchmarking over {num_iterations} iterations...")
    latencies = []
    
    for _ in range(num_iterations):
        start_time = time.perf_counter()
        compiled_model([input_tensor_img, input_tensor_lang])
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000) # in ms
        
    avg_latency = np.mean(latencies)
    throughput = 1000.0 / avg_latency
    
    print("\n--- Benchmark Results ---")
    print(f"Target Device: {device_name}")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"Throughput: {throughput:.2f} FPS")
    print("-------------------------\n")

if __name__ == "__main__":
    benchmark(device_name="AUTO")
    # Uncomment to explicitly test specific devices if available on the Intel Core Ultra
    # benchmark(device_name="CPU")
    # benchmark(device_name="GPU")
    # benchmark(device_name="NPU")

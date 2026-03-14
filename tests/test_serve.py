import asyncio
import time
import aiohttp

# Ray Serve binds to 8000 by default
URL = "http://127.0.0.1:8000/predict"

async def fire_request(session: aiohttp.ClientSession, index: int):
    """Sends a single HTTP POST request to the inference endpoint."""
    payload = {
        "title": f"Test Article {index}: Transformer Models in PyTorch",
        "description": "Analyzing deep learning and attention mechanisms."
    }
    
    # Start the timer for this specific request
    req_start = time.time()
    
    # We await the response. If the server queues this, execution pauses here.
    async with session.post(URL, json=payload) as response:
        result = await response.json()
        
        # Stop the timer the millisecond the response arrives
        req_end = time.time()
        latency = req_end - req_start
        
        return index, result, latency

async def main():
    num_requests = 200
    print(f"Initiating load test: Firing {num_requests} concurrent requests...")
    
    # Start the global timer
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        # Create pending tasks immediately, without waiting for the first to finish
        tasks = [fire_request(session, i) for i in range(num_requests)]
        
        # Gather executes them concurrently
        results = await asyncio.gather(*tasks)
        
    # Stop the global timer
    end_time = time.time()
    total_time = end_time - start_time
    
    # Extract latencies to calculate averages
    latencies = [res[2] for res in results]
    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)
    min_latency = min(latencies)
    
    # Calculate overall throughput
    rps = num_requests / total_time
    
    print(f"\n--- Load Test Results ---")
    print(f"Total Time: {total_time:.4f} seconds")
    print(f"Throughput: {rps:.2f} Requests Per Second (RPS)")
    print(f"Average Latency: {avg_latency:.4f} seconds")
    print(f"Min/Max Latency: {min_latency:.4f}s / {max_latency:.4f}s")
    
    print("\nSample Response (from request 0):")
    for idx, res, lat in results:
        if idx == 0:
            print(f"Latency: {lat:.4f}s | Result: {res}")
            break

if __name__ == "__main__":
    asyncio.run(main())
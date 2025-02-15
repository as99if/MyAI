import asyncio

async def a():
  """Simulates an asynchronous operation that returns an integer."""
  await asyncio.sleep(1)  # Simulate some work being done
  return 10  # Example integer value

async def b():
  """Simulates another asynchronous operation that returns an integer."""
  await asyncio.sleep(2)  # Simulate some work being done
  return 20  # Example integer value

async def z():
  """Executes a and b concurrently and sums their results."""
  task_a = asyncio.create_task(a())  # Create a task for a()
  task_b = asyncio.create_task(b())  # Create a task for b()

  # Wait for both tasks to complete and get their results
  result_a = await task_a
  result_b = await task_b

  return result_a + result_b

async def main():
  """Runs the asynchronous calculation."""
  result = await z()
  print(f"The result of a + b is: {result}")

if __name__ == "__main__":
  asyncio.run(main())
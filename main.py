from mlxserver import MLXServer
import uvicorn
 

server = MLXServer(model="mlx-community/dolphin-2.6-mistral-7b-dpo-laser-4bit-mlx")

uvicorn.run(server.app, host="localhost", port=9999)


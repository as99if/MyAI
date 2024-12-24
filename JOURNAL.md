https://github.com/armbues/SiLLM

Install in venv

        python3.11 -m venv .venv
        source .venv/bin/activate

Install SiLLM for MLX inference, fine/tuning

        git clone https://github.com/armbues/SiLLM.git
        cd SiLLM/app
        pip install -r requirements.txt
        python -m chainlit run app.py -w



Roadmap:
1. Chat server and UI
1.1. Chat history, multiple chat.. reference folder
2. RAG
3. File upload in chat session, Camera input
4. Chat history
5. Autogen - two AI client (agent) chatting or thinking
6. Fine tuning (Lora) server
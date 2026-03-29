# Financial Analysis AI Agent

**Status: Ongoing / Active Development**

## Overview
A Python-based AI agent designed to automate financial research. The agent integrates large language models with external financial APIs to query real-time market data, process quantitative metrics, and generate concise portfolio summaries. 

## Features
* **Real-Time Data Retrieval:** Utilizes function calling to fetch live stock prices and historical market data.
* **Data Structuring:** Cleans and processes raw financial data using Pandas before passing it to the reasoning engine.
* **Autonomous Reasoning:** Leverages LangChain and OpenAI's models to interpret user queries and execute the appropriate data-gathering tools.

## Tech Stack
* Python
* LangChain & OpenAI API
* Pandas
* yfinance (Market Data)
* SEC EDGAR API (Planned integration for 10-K/10-Q parsing)

## Setup Instructions
1. Clone the repository.
2. Install the required dependencies: `pip install -r requirements.txt`
3. Create a `.env` file in the root directory and add your OpenAI API key: `OPENAI_API_KEY="your_api_key_here"`
4. Run the agent: `python main.py`

[cite_start]**Author:** Dev Goyal [cite: 1]
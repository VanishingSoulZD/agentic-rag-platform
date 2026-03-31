def calculator(expression: str) -> str:
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"error: {str(e)}"


from langchain.tools import Tool
calculator_tool = Tool(
    name="Calculator",
    func=calculator,
    description="Useful for math calculations. Input should be a valid math expression."
)


def get_weather(city: str) -> str:
    """Get weather for a given city"""

    mock_data = {
        "台北": "晴天 25°C",
        "Taipei": "晴天 25°C",
        "北京": "多云 18°C",
        "上海": "小雨 20°C"
    }

    return mock_data.get(city, f"{city}天气未知")

weather_tool = Tool(
    name="Weather",
    func=get_weather,
    description="Get weather information for a city. Input should be a city name."
)

import sqlite3
def query_db(query: str) -> str:
    """Execute SQL query on users database"""

    try:
        conn = sqlite3.connect("test.db")
        cursor = conn.cursor()

        cursor.execute(query)
        results = cursor.fetchall()

        conn.close()

        return str(results)

    except Exception as e:
        return f"error: {str(e)}"
db_tool = Tool(
    name="Database",
    func=query_db,
    description="Query user database. Input should be a SQL query."
)

from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
llm = ChatOpenAI(model="accounts/fireworks/models/deepseek-v3p1", temperature=0)

def my_agent():
    agent = initialize_agent(
        tools=[calculator_tool, weather_tool, db_tool],
        llm=llm,
        agent="zero-shot-react-description",
        verbose=True
    )
    result = agent.run("What is 1 * 7 + 5?")
    print(result)
    result = agent.run("今天台北天气怎么样？?")
    print(result)
    agent.run("查询 users 表中所有数据")
    print(result)

import json

def planner(question: str):
    prompt = f"""
You are a planner.

Break the question into steps in JSON format.

Question:
{question}

Return format:
[
  "step1",
  "step2",
  ...
]
"""
    response = llm.predict(prompt)

    try:
        steps = json.loads(response)
    except:
        steps = [response]

    return steps

import re
tools = [calculator_tool, weather_tool, db_tool]
def run_tool(tool_name, tool_input):
    for tool in tools:
        if tool.name == tool_name:
            return tool.func(tool_input)
    return "Tool not found"

# ReAct + re
def executor(question: str, max_steps=5):
    prompt = f"""
You are an agent.

You have access to tools:
{[tool.name for tool in tools]}

Follow this format:

Thought: ...
Action: tool_name
Action Input: input
Observation: result
... (repeat)
Final Answer: ...

Question: {question}
"""

    history = prompt

    for _ in range(max_steps):
        response = llm.predict(history)

        print("=== LLM ===")
        print(response)

        # 判断是否结束
        if "Final Answer" in response:
            return response

        # 解析 Action
        action_match = re.search(r"Action:\s*(.*)", response)
        input_match = re.search(r"Action Input:\s*(.*)", response)

        if not action_match or not input_match:
            break

        tool_name = action_match.group(1).strip()
        tool_input = input_match.group(1).strip()

        observation = run_tool(tool_name, tool_input)

        # 回填 observation
        history += f"\n{response}\nObservation: {observation}\n"

    return history

from langchain.schema import SystemMessage, HumanMessage, AIMessage

class ConversationMemory:
    def __init__(self, max_turns=5):
        self.history = []
        self.max_turns = max_turns

    def add(self, role, content):
        self.history.append((role, content))

        # 控制长度（非常重要）
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-self.max_turns * 2:]

    def get(self):
        return self.history

def build_messages(memory, question):
    messages = [SystemMessage(content="You are an agent with tools.")]

    # 加历史
    for role, content in memory.get():
        if role == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))

    return messages

# ReAct + tool_calling
def agent_loop(question, memory, max_steps=5):
    # ✅ 先写 user
    memory.add("user", question)
    messages = build_messages(memory, question)

    for _ in range(max_steps):
        response = llm(messages)
        print(response)
        # 写入 memory（关键）
        memory.add("assistant", response.content)

        # 👉 尝试解析 tool_calls（关键）
        if hasattr(response, "additional_kwargs"):
            tool_calls = response.additional_kwargs.get("tool_calls")

            if tool_calls:
                for call in tool_calls:
                    name = call["function"]["name"]
                    args = json.loads(call["function"]["arguments"])

                    observation = run_tool(name, args.get("input") or args.get("expression") or "")

                    messages.append({
                        "role": "assistant",
                        "content": str(response.content)
                    })

                    messages.append({
                        "role": "tool",
                        "name": name,
                        "content": observation
                    })

                    break
                continue

        # 如果没有 tool_calls → 认为是最终答案
        return response.content

    return "Max steps reached"

def summarize(question: str, executor_output: str):
    prompt = f"""
Question:
{question}

Agent Output:
{executor_output}

Please give a clean final answer.
"""
    return llm.predict(prompt)


def run_agent(question: str):
    print("=== Planner ===")
    steps = planner(question)
    print(steps)

    print("\n=== Executor ===")
    # executor_output = executor(question)
    memory = ConversationMemory()
    executor_output = agent_loop(question, memory)

    print("\n=== Summary ===")
    final = summarize(question, executor_output)
    print(final)
    return final

if __name__ == "__main__":
    # run_agent("查台北天气，然后把温度乘以2并总结")
    run_agent("查上海天气，并计算温度加10")
    # run_agent("如果北京温度是当前天气温度，计算温度*3并说明结果")




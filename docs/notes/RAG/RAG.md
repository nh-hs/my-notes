---
title: RAG与LangChain
tags:
  - RAG
---

# RAG与LangChain

## 一、RAG概述

### 1. RAG的基本原理

![image-20260822202524012](.\assets\image-20260822202524012.png)

### 2. RAG与微调

![image-20260822204929442](.\assets\image-20260822204929442.png)

## 二、LangChain简介

### 1. ollama

#### 1.1 在python中使用Ollama API

~~~python
import ollama

# 聊天式
# response = ollama.chat(model='deepseek-r1:1.5b',
#                        messages=[{'role': 'user', 'content': '为什么天空是蓝色的？', }])

# 生成式
# response = ollama.generate(model='deepseek-r1:1.5b', prompt='为什么天空是蓝色的？')

# print(response)
# print(response['message']['content'])

# 流式
stream = ollama.chat(
    model='deepseek-r1:1.5b',
    messages=[{'role': 'user', 'content': '为什么天空是蓝色的？'}],
    stream=True,
)

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
~~~

### 2. LangChain基础知识

#### 2.1 概述

> LangChain由 Harrison Chase 创建于2022年10月

> **主要组件**：
>
> ![image-20260822204545481](.\assets\image-20260822204545481.png)
>
> ![2](.\assets\2.png)

> **核心包**：
>
> - **langchain-core**：聊天模型和其他组件的基础抽象
> - **集成包（例如 langchain-openai、langchain-anthropic 等）**：重要的集成被拆分为轻量级的独立包，由 LangChain 团队和集成方共同维护
> - **langchain**：包含链（chains）、智能体（agents）和检索策略，这些构成了应用的认知架构
> - **langchain-community**：由社区维护的第三方集成
> - **langgraph**：一个编排框架，用于将 LangChain 组件组合成可用于生产的应用，支持持久化、流式处理及其他关键特性

#### 2.2 Models

> LangChain模型组件提供了与各种模型的集成，并为所有模型提供一个精简的**统一接口**
>
> LangChain目前支持三种类型的模型：LLMs、Chat Models(聊天模型)、Embeddings Models(嵌入模型）

##### 2.2.1 LLMS

> 大语言模型接收文本字符作为输入，返回的也是文本字符 -> 一般只有一轮

- 通过ChatOpenAI进行调用

  ~~~python
  from langchain_openai import ChatOpenAI
  
  # 实例化模型调用
  model = ChatOpenAI(model="qwen3.7-flash-2026-07-15",
                     openai_api_base="https://ws-7fxi94n23wuxy7l1.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
                     temperature=0.9)
  res = model.invoke(["你和deepseek-v4-flash谁更厉害"])
  print(res.content)
  # print(res.text)  # langchain自定义的文本输出
  ~~~

- 通过Ollama进行调用Ollama支持模型

  ~~~python
  from langchain_ollama import OllamaLLM
  
  model = OllamaLLM(model="qwen2:1.5b")
  # 获取问答结果
  res = model.invoke("为什么天空是蓝色的？")
  print(res)
  ~~~

##### 2.2.2 Chat Models(聊天模型)

> 聊天消息包含下面几种类型，使用时需要按照约定传入合适的值：
>
> - AIMessage:AI 输出的消息，可以是针对问题的回答
> - HumanMessage:用户信息，由人给出的信息发送给LLMs的提示信息
> - SystemMessage:用于指定模型具体所处的环境和背景，如角色扮演等
> - ChatMessage:Chat 消息可以接受任意角色的参数

~~~python
from langchain_ollama import ChatOllama
# 不同消息的封装对象AI，HUMAN，SYSTEM
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# 实例化模型
model = ChatOllama(model="qwen2:1.5b")
message = [
    SystemMessage(content="你是一个科普助手"),
]
# 构建消息列表
while True:
    user = input("请输入问题：")
    if user == "exit":
        break
    # 追加用户问题
    message.append(HumanMessage(content=user))
    res = model.invoke(message)
    print(res.content)
    # 追加AI的回答
    message.append(AIMessage(content=res.content))
~~~

##### 2.2.3 Embeddings Models(嵌入模型)

> 将字符串作为输入，返回一个浮动数的列表，可以为文本创建向量映射。在NLP中，Embedding的作用是将数据向量化

> 主流Embedding模型的对比：
>
> - beg-m3（主要使用） -> 多语言、多功能、多粒度Embedding模型，1024维
> - text-embedding-ada-002 -> 支持多种语言，但对中文等亚洲语言的支持相对较弱，1536维
> - bge-large-zh -> 中文为主，英文支持一般，最大输入长度短，1024维
> - multilingual-e5-large -> 对多语言都有较好的支持，最大输入长度短，1024维
> - mxbai-embed-large -> 多语言（主要英文），1024维

~~~python
from langchain_ollama import OllamaEmbeddings  # ollama
from langchain_openai import OpenAIEmbeddings  # 百炼

# 初始化Ollama嵌入模型，使用bge-m3模型
model = OllamaEmbeddings(model="bge-m3")
# 百炼平台
# model = OpenAIEmbeddings(model="qwen3.7-text-embedding",
#                          openai_api_base="https://ws-7fxi94n23wuxy7l1.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
#                          check_embedding_ctx_length=False  # 关键！禁止转成token列表
#                          )

# 对单个查询文本进行嵌入编码
res1 = model.embed_query('这是第一个测试文档')

print(f'result1->{res1}')
print(f'result1的长度->{len(res1)}')  # 维度 1024维

# 对多个文档进行批量嵌入编码
res2 = model.embed_documents(['这是第一个测试文档', '这是第二个测试文档'])
print(res2)
~~~

#### 2.3 Prompts

> Prompt是指当用户输入信息给模型时加入的提示

##### 2.3.1 通用prompt

- zero-shot提示方式

  > 1. 定义模板
  > 2. 模板初始化
  > 3. 模板填充

  ~~~python
  from langchain_core.prompts import PromptTemplate
  from langchain_ollama import OllamaLLM
  
  # 实例化模型
  model = OllamaLLM(model="qwen2.5:7b")
  # 定义模板
  template = """
      我的邻居姓{lastname}
      他生了个{sex}，给他的孩子起个名字
      """
  # 模板初始化
  prompt = PromptTemplate(
      input_variables=["lastname", "sex"],
      template=template
  )
  # 模板的填充
  prompt_text = prompt.format(lastname="范", sex="女孩")
  print(prompt_text)
  # 调用模型
  res = model.invoke(prompt_text)
  print(res)
  ~~~

- few-shot提示方式

  > 将样例模板化

  ~~~python
  # from langchain import PromptTemplate, FewShotPromptTemplate # Langchain 0.x版本使用
  from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
  from langchain_ollama import OllamaLLM
  
  model = OllamaLLM(model="qwen2.5:7b")
  
  # 需要填入的样例
  examples = [
      {"word": "开心", "antonym": "难过"},
      {"word": "高", "antonym": "矮"}]
  
  # 样例模板
  example_template = """
  单词: {word}
  反义词: {antonym}\\n
  """
  
  # 初始化样例模板
  example_prompt = PromptTemplate(
      input_variables=["word", "antonym"],
      template=example_template,
  )
  
  few_shot_prompt = FewShotPromptTemplate(
      examples=examples,
      example_prompt=example_prompt,
      prefix="给出每个单词的反义词",
      suffix="单词: {input}\\n反义词:",
      input_variables=["input"],
      example_separator="\\n",
  )
  # 查看最终提示词样子
  print(few_shot_prompt.format(input="开心"))
  # 调用模型
  prompt_text = few_shot_prompt.format(input="粗")
  
  print(model.invoke(prompt_text))
  ~~~

##### 2.3.2 ChatPrompts

适合交互式对话应用，如聊天机器人、智能客服等，这些应用需要处理用户和LLM之间的多轮对话。

> ChatPromptTemplate
>
> SystemMessagePromptTemplate
>
> HumanMessagePromptTemplate
>
> history=[("system","......"),('human',"......"),("ai","......")]

>初始化提示词模板：
>
>  1. from_template：初始化的是**字符串**消息体 ->  初始化成HumanMessage类型的消息列表
>  2. from_messages：初始化的是**列表**消息体
>
>填充提示词模板
>
>  format_messages：填充消息提示词列

- 直接提问

  ~~~python
  from langchain_core.prompts import ChatPromptTemplate
  from langchain_ollama import OllamaLLM
  
  # 实例化模型
  model = OllamaLLM(model="qwen2.5:7b")
  
  # 定义提示词模版
  template_str = "帮我讲个关于{name}笑话吧"
  prompt_template = ChatPromptTemplate.from_template(template_str)
  prompt = prompt_template.format_messages(name="气球")
  print(f'prompt-->{prompt}')
  
  # 调用模型
  result = model.invoke(prompt)
  print(f'result-->{result}')
  ~~~

- zero-shot提示方法

  ~~~python
  from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
  from langchain_core.messages import SystemMessage
  from langchain_ollama import ChatOllama
  
  # 实例化模型
  model = ChatOllama(model="qwen2.5:7b")
  
  messages = [
      SystemMessage("你是取名专家。"),
      HumanMessagePromptTemplate.from_template('我的邻居姓{lastname}，他生了个儿子，给他儿子起个名字。')
  ]
  # 会话消息模板构建
  chat_template = ChatPromptTemplate.from_messages(messages)
  # 填充会话消息模板
  res = chat_template.format_messages(lastname='王')
  print(res)
  
  # 调用模型
  result = model.invoke(res)  # 返回结果除了content外，还有元数据信息
  print(f'result-->{result}')
  ~~~

- few-shot提示方法

  > 基于MessagesPlaceholder（消息插槽）来进行模版填充

  ~~~python
  # MessagesPlaceholder 消息插槽
  from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
  from langchain_ollama import ChatOllama
  from langchain_core.messages import SystemMessage
  
  model = ChatOllama(model="qwen2.5:7b")
  
  # 创建prompt模板
  prompt_template = ChatPromptTemplate.from_messages([
      SystemMessage("给出每个单词的反义词"),
      # 消息插槽
      MessagesPlaceholder('history'),
      HumanMessagePromptTemplate.from_template('{question}')
  ])
  # 提示词填充
  # Use one of 'human', 'user', 'ai', 'assistant', 'function', 'tool', 'system', or 'developer'
  history = [("human", "开心"), ("ai", "难过"), ("human", "高"), ("ai", "矮")]
  prompt = prompt_template.format_prompt(history=history, question="小")
  print(prompt)
  # 模型调用
  result = model.invoke(prompt)
  print(result.content)
  ~~~

#### 2.4 Chains

> Chains描述了将LLM与其他**组件**结合起来完成一个应用程序的过程，主要使用**LCEL**方法
>
> LECL(Lang Chain Expression Language)：
>
> ​	语法规则：使用`|`符号将不同的组件连接起来，形成一个链式结构
>
> ​	注：上一个的输出作为下一个的输入，输出和输入的类型必须保持一致，否则不能连接

- 单链

  ~~~python
  from langchain_core.prompts import PromptTemplate
  from langchain_ollama import OllamaLLM
  import os
  
  os.environ["LANGCHAIN_TRACING_V2"] = "true"  # 启用 LangSmith链路追踪
  os.environ["LANGSMITH_API_KEY"] = "lsv2_pt_xxxxxxxxxxxxxxxx"
  os.environ["LANGCHAIN_PROJECT"] = "人民的好邻居"  # 项目名称
  
  model = OllamaLLM(model="qwen2.5:7b")
  
  template = "我的邻居姓{lastname},他生了一个{sex}，请给他孩子起个名字"
  prompt = PromptTemplate(
      input_variables=["lastname", "sex"],
      template=template
  )
  
  # 前一个的输出作为后一个的输入
  chain = prompt | model
  # 用字典形式传参
  res = chain.invoke({"lastname": "李", "sex": "哪吒"})  
  print(res)
  ~~~

- 多链

  ~~~python
  from langchain_core.prompts import PromptTemplate
  from langchain_ollama import OllamaLLM
  import os
  
  os.environ["LANGCHAIN_TRACING_V2"] = "true"  # 启用 LangSmith链路追踪
  os.environ["LANGSMITH_API_KEY"] = "lsv2_pt_xxxxxxxxxxxxxxxx"
  os.environ["LANGCHAIN_PROJECT"] = "人民的好邻居"  # 项目名称
  
  model = OllamaLLM(model="qwen2.5:7b")
  
  template = "我的邻居姓{lastname},他生了一个{sex}，请给他孩子起个名字，只返回一个名字，除此之外不要有其他内容"
  prompt = PromptTemplate(
      input_variables=["lastname", "sex"],
      template=template
  )
  
  first_chain = prompt | model
  
  # 创建第二条链
  second_template = "邻居的孩子叫{child_name},请给他起个小名"
  second_prompt = PromptTemplate(
      input_variables=["child_name"],
      template=second_template
  )
  second_chain = second_prompt | model
  
  # 连接两条链
  overall_chain = first_chain | second_chain
  # 执行时只用传第一个链的参数
  res = overall_chain.invoke({"lastname": "李", "sex": "女孩"})
  print(res)
  ~~~

##### 2.4.0 langsmith

> ```python
> import os
> 
> os.environ["LANGCHAIN_TRACING_V2"] = "true"  # 启用 LangSmith链路追踪
> os.environ["LANGSMITH_API_KEY"] = "你的API"
> os.environ["LANGCHAIN_PROJECT"] = "人民的好邻居"  # 项目名称
> ```

> 作用：
>
> 1. 链路追踪 Tracing（最核心）
> 2. 监控 Monitoring（生产环境）
> 3. 评估 Evaluation（LLM 质量测试）
> 4. Prompt 工程与版本管理
> 5. Agent / 应用部署与管理（新功能）

#### 2.5 output_parsers（输出解析器）

> LLM 的输出是**自然语言文本**，LangChain 输出解析器负责获取 LLM 的输出并将其转换为更合适的格式

>部分解析器：
>
>- StrOutputParser：默认解析器。将 LLM 的输出直接解析为字符串。
>- CommaSeparatedListParser：将 LLM 输出的内容用**逗号**分隔的文本解析为列表。
>- JsonOutputParser：极其常用。将 LLM 输出的**JSON 字符串**解析为**Python 字典**。
>- PydanticOutputParser：极其常用。将 LLM 输出解析为预先定义的** Pydantic 对象**，提供类型安全和数据验证。
>- DatetimeOutputParser： 从文本中智能地解析出日期和时间信息。

> 基本步骤：
>
> 1. 导入对应解析包
> 2. 实例化解析器对象
> 3. 获得结构化结果输出约束的提示词(get_format_instructions())
> 4. 将结果约束的提示词填充到提示词模板中
> 5. 以chain的形式调用结果解析器对象

> 实现方法：
>
> 1. get_format_instructions方法：提供格式指导（提示词），约束模型按要求输出
> 2. parse方法：将模型输出文本类型转换成需要的类型

##### 2.5.1 字符串解析器（StrOutputParser）

~~~python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

model = ChatOllama(model="qwen2.5:7b")
# 实例化字符串解释器
parser = StrOutputParser()
prompt = PromptTemplate.from_template("请将{text}翻译成英文")
chain = prompt | model | parser  # 放在最后
res = chain.invoke({"text": "我叫张三"})
print(res)
~~~

##### 2.5.2 列表解析器（CommaSeparatedListOutputParser）

> 将逗号分隔的文本转换为Python列表

~~~python
# 列表解析器：将逗号分隔的文本转换为Python列表
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

model = OllamaLLM(model="qwen2.5:7b")
# 实例化列表解释器
parser = CommaSeparatedListOutputParser()

# 创建带格式说明的模板
# 获得结构化结果输出约束的提示词
format_instructions = parser.get_format_instructions()
prompt = ChatPromptTemplate.from_template(
    "用中文列出{topic}的五个最重要特点。\n{format_instructions}")
# 组合组件
chain = prompt | model | parser

# 调用链
result = chain.invoke({
    "topic": "大模型",
    "format_instructions": format_instructions
})
print(result)
~~~

##### 2.5.3 JSON解析器(JsonOutputParser)

> 将JSON格式文本转换为Python字典或列表

~~~python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"  # 启用 LangSmith链路追踪
os.environ["LANGSMITH_API_KEY"] = "lsv2_pt_xxxxxxxxxxxxxxxx"
os.environ["LANGCHAIN_PROJECT"] = "人民的好邻居"  # 项目名称
# 实例化模型
model = OllamaLLM(model="qwen2.5:7b")

# 创建JSON解析器
json_parser = JsonOutputParser()

# 创建带格式说明的提示模板
json_format_instructions = json_parser.get_format_instructions()
json_prompt = ChatPromptTemplate.from_template(
    "生成一个包含{person}基本信息的JSON。应包括姓名、职业、年龄和技能列表, 不要包含任何注释或额外说明。\n{format_instructions}"
)

# 组合组件
json_chain = json_prompt | model | json_parser

# 调用链
result = json_chain.invoke({
    "person": "雷军",
    "format_instructions": json_format_instructions
})

print(result)
~~~

##### 2.5.4 Pydantic解析器(PydanticOutputParser)

> 使用Pydantic模型定义输出结构:
>
> class Movie(BaseModel):
>     title: str = Field(description="电影标题")
>     director: str = Field(description="导演姓名")
>     year: int = Field(description="上映年份")
>     genre: List[str] = Field(description="电影类型")
>     rating: float = Field(description="评分（1-10）")

~~~python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from langchain_ollama import OllamaLLM

# 实例化模型
model = OllamaLLM(model="qwen2.5:7b")


# 定义Pydantic模型
class Movie(BaseModel):
    title: str = Field(description="电影标题")
    director: str = Field(description="导演姓名")
    year: int = Field(description="上映年份")
    genre: List[str] = Field(description="电影类型")
    rating: float = Field(description="评分（1-10）")


# 创建Pydantic解析器
pydantic_parser = PydanticOutputParser(pydantic_object=Movie)

# 创建带格式说明的提示模板
format_instructions = pydantic_parser.get_format_instructions()
pydantic_prompt = ChatPromptTemplate.from_template(
    "生成一部{genre}电影的信息。\n{format_instructions}"
)

# 组合组件
pydantic_chain = pydantic_prompt | model | pydantic_parser

# 调用链
movie_data = pydantic_chain.invoke({
    "genre": "冒险",
    "format_instructions": format_instructions
})
print(movie_data)
# 运行结果：title='神秘岛屿探险' director='李晓明' year=2024 genre=['冒险', '科幻', '惊悚'] rating=8.5
~~~

##### 2.5.5 自定义解析器

> 1. 定义类继承BaseOutputParser
> 2. 重写parse方法（将大模型输出的字符串格式转为需要的格式）
> 3. 重写get_format_instructions方法（需要的提示词）

~~~python
from langchain_core.output_parsers import BaseOutputParser
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

# 实例化模型
model = OllamaLLM(model="qwen2.5:7b")


class CustomKeyValueParser(BaseOutputParser[Dict[str, Any]]):
    """解析形如"key:value"的文本"""

    # 重写parse方法（将大模型输出的字符串格式转为json格式）
    def parse(self, text: str) -> Dict[str, Any]:
        """从文本中解析键值对"""
        result = {}
        lines = text.strip().split('\n')

        for line in lines:
            if ':' in line:
                # 把字符串line只在第一个冒号`:`的位置切开，分成两部分，左边赋值给key，右边赋值给value。
                # 参数：sep表示按什么切，maxsplit = 1表示只切一次（第一个冒号）
                key, value = line.split(':', 1)
                # 把清理过后的 `key` 和 `value`，存入字典 `result`
                result[key.strip()] = value.strip()

        return result

    def get_format_instructions(self) -> str:
        """提供格式指导给模型"""
        return """请以'键: 值'的格式返回信息，每行一个键值对。
                例如：
                名称: 爱因斯坦
                职业: 物理学家
                贡献: 相对论"""


# 使用自定义解析器
custom_parser = CustomKeyValueParser()
custom_prompt = ChatPromptTemplate.from_template(
    "提供关于{person}的基本信息。\n{format_instructions}"
)

# 组合组件
custom_chain = custom_prompt | model | custom_parser

# 调用模型
result = custom_chain.invoke({
    "person": "屠呦呦",
    "format_instructions": custom_parser.get_format_instructions()
})
print(result)
~~~

#### 2.6 Memory

> 大模型本身不具备上下文的概念，因此langchain提供了Memory组件, Memory分为两种类型：**短期记忆和长期记忆** 
>
> 短期记忆：一般指单一会话时传递数据
>
> 长期记忆：处理多个会话时获取和更新信息

> 使用ChatMessageHistory手动添加上下文
>
> 1. 将历史会话消息以对象形式储存
> 2. 以文件形式存储json数据
> 3. 读取json数据文件
> 4. 将读取到的json数据转消息对象

~~~python
# messages_to_dict：消息转字典类型  messages_from_dict：从字典类型恢复消息
from langchain_core.messages import messages_to_dict, messages_from_dict, SystemMessage
import json
from langchain_community.chat_message_histories import ChatMessageHistory

# 1、将历史会话消息以对象形式储存
history = ChatMessageHistory()
# 组装演示数据
history.add_message(SystemMessage("你是一个吊毛，无论别人说什么，你只会回复'吊毛'两个字"))
history.add_user_message("hello")
history.add_ai_message("吊毛")
print(history.messages)

# 2、以文件形式存储json数据
# 消息转字典
msgs_to_dict = messages_to_dict(history.messages)
print(msgs_to_dict)
with open("./data/history.json", "w", encoding="utf-8") as f:
    # ensure_ascii=False，中文不会被转码 indent=4，每一层级前加4个空格
    f.write(json.dumps(msgs_to_dict, ensure_ascii=False, indent=4))

# 3、读取json数据文件
dicts = json.load(open("./data/history.json", "r", encoding="utf-8"))
print(dicts)

# 4、将读取到的json数据转消息对象
msgs_from_dict = messages_from_dict(dicts)
print(msgs_from_dict)

~~~

##### 2.6.1 短期记忆

> 短期记忆 =  Agent 状态（State） + 检查点持久化（Checkpointer） + 线程标识（thread_id）

> - 状态(State)：通常是一个包含 messages 字段的字典或 Pydantic 模型（如 MessagesState 或自定义 CustomState），用于存储当前会话的所有消息、中间变量、工具调用结果等
> - 检查点器(Checkpointer)：负责将状态序列化并持久化到内存、SQLite、PostgreSQL 等后端。每次状态变更（如新增一条消息）都会触发一次检查点保存
> - 会话ID(thread_id)：作为会话的唯一标识符，确保不同用户或不同对话之间的状态完全隔离。即使多个用户并发交互，也不会发生记忆混淆

> 基于agent来演示短期记忆
> 1. 创建一个agent对象
> 2. 定义一个会话id
> 3. 构建一个检查的对象，构建一个内存对象给检查点进行管理（及时更新内存的数据）
> 4. 使用agent对象来基于会话id和检查点对象实现多轮对话

~~~python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

# 初始化大模型，这里使用的是 Qwen 的聊天模型
model = ChatOllama(model="qwen2.5:7b")
# 1、构建一个检查的对象，构建一个内存对象交给检查点进行管理（及时更新内存的数据）
checkpoint = InMemorySaver()
# 2、创建一个agent对象
agent = create_agent(model=model, checkpointer=checkpoint, system_prompt="你是一个翻译官，擅长中英互译。")
# 3、定义一个会话id
config = {
    "configurable": {"thread_id": "123"}
}
input_1 = {
    "messages": [HumanMessage("你好，我叫小呆。")]
}

res1 = agent.invoke(config=config, input=input_1)
print(res1)

# 问之前问题，看能不能回答正确
input_2 = {
    "messages": [HumanMessage("我叫什么名字")]
}

res2 = agent.invoke(config=config, input=input_2)
print(res2)

# 跨会话短期记忆,短期记忆不能夸会话
config2 = {
    "configurable": {"thread_id": "456"}
}
res3 = agent.invoke(config=config2, input=input_2)
print(res3)
~~~

##### 2.6.2 长期记忆

> 对于需要长期运行和可靠记忆的应用，推荐使用数据库进行持久化

#### 2.7 Indexes

> Indexes组件的目的是让LangChain具备处理文档处理的能力，主要包含：
>
> - 文档加载器
> - 文本分割器
> - VectorStores
> - 检索器

##### 2.7.1 文档加载器

> 主要基于`Unstructured` 包，文档加载器使用起来很简单，只需要引入相应的loader工具
>
> LangChain支持的文档加载器 (部分)：
>
> | 文档加载器           | 描述               |
> | :------------------- | :----------------- |
> | CSV                  | 加载 CSV 文件      |
> | JSON Files           | 加载 JSON 文件     |
> | Jupyter Notebook     | 加载 notebook 文件 |
> | Markdown             | 加载 markdown 文件 |
> | Microsoft PowerPoint | 加载 ppt 文件      |
> | PDF                  | 加载 pdf 文件      |
> | Images               | 加载图片           |
> | File Directory       | 加载目录下所有文件 |
> | HTML                 | 网页               |

- 通用型文件解析器（UnstructuredLoader）

  > 处理各种**非结构化**或**半结构化**文件，例如 PDF、Word (.docx)、PowerPoint、HTML 等

  ~~~python
  from langchain_unstructured import UnstructuredLoader
  
  # 创建 UnstructuredLoader 对象
  loader = UnstructuredLoader(file_path='./data/衣服属性清单.txt', encoding='utf8')
  docs = loader.load()
  print(f'docs-->{docs}')
  print(f'len-->{len(docs)}')
  print(f'第一行数据-->{docs[0].page_content}')
  ~~~

- 纯文本文件加载器（TextLoader）

  > 处理**纯文本**格式的文件。它不进行任何格式解析，只是将文件内容原样读取为文本,如 .txt、.md、.csv 等

  ~~~python
  from langchain_community.document_loaders import TextLoader
  
  # 创建 TextLoader 对象
  loader = TextLoader('./data/衣服属性清单.txt', encoding='utf8')
  docs = loader.load()
  print(f'docs-->{docs}')
  print(f'len-->{len(docs)}')
  print('第一行数据-->{}'.format(docs[0].page_content.split('\n')[0]))
  ~~~

##### 2.7.2 文档分割器

> LangChain中最基本的文本分割器是`CharacterTextSplitter`，LangChain还支持其他文档分割器 (部分)：

| 分割器名称                     | 功能描述                                                     | 类型           | 工业场景应用                                              |
| :----------------------------- | :----------------------------------------------------------- | :------------- | :-------------------------------------------------------- |
| CharacterTextSplitter          | 简单按指定分隔符（如换行、逗号）直接分割。                   | 基础字符解析   | 简单字符串或 CSV 数据处理，如传感器数据日志。             |
| RecursiveCharacterTextSplitter | 递归按字符分割，先尝试自然边界（如段落、句子），太大则继续细分。 | 通用字符解析   | 通用文本处理，如日志、报告、PDF 文档分割，便于 RAG 检索。 |
| TokenTextSplitter              | 按 token（词元）分割，支持 LLM token 计数。                  | 基于Token 解析 | LLM 输入优化，如处理 API 响应或长查询，控制 token 限制。  |
| SentenceTextSplitter           | 按句子边界分割，使用 NLP 识别句子（包括标点）。              | 语义解析       | 自然语言文本，如文章或对话分析，保持句子完整。            |
| SpacyTextSplitter              | 使用 SpaCy NLP 库按句子或实体分割（需安装 SpaCy）。          | 语义解析       | 高级 NLP 场景，如实体提取或生物医学文本。                 |
| NLTKTextSplitter               | 使用 NLTK 库按句子或词分割（需安装 NLTK）。                  | 语义解析       | 文本研究或分析，如时间序列数据描述。                      |
| MarkdownHeaderTextSplitter     | 按 Markdown 结构（如标题、列表）智能分割。                   | 结构化解析     | Markdown 文档分割，保留语义结构，用于知识库构建。         |
| HTMLSplitter                   | 按 HTML 标签（如 、）分割网页内容。                          | 结构化解析     | 网页数据爬取，如在线技术文档或新闻提取。                  |
| LatexTextSplitter              | 按 LaTeX 结构（如章节、公式）分割。                          | 结构化解析     | 学术论文或数学文档处理。                                  |
| PythonCodeTextSplitter         | 按 Python 代码结构（如函数、类）分割。                       | 代码解析       | 源代码文件分析，如脚本调试或代码库管理。                  |

- CharacterTextSplitter（按指定分隔符划分）

  > 参数：separator：按什么字符分割  chunk_size：分割最大长度  chunk_overlap：分割后重叠的长度
  >
  > 分割原则：
  >
  > 1. 优先用 separator 把全文先切为小段
  > 2. 组装文本块，累积到接近 chunk_size，如果加上下一个片段后，总长度**超过 chunk_size**：
  >    - 当前块完成，输出；
  >    - 开启新块；**新块开头带上上一个块末尾 chunk_overlap 长度的字符，实现重叠**。
  > 3. 如果单个片段本身就大于 chunk_size，会直接整块输出

  ~~~python
  from langchain_core.documents import Document
  # CharacterTextSplitter：支持固定分块大小分割，也支持特殊字符分割
  from langchain_text_splitters import CharacterTextSplitter
  
  # 实例化分割器
  # separator：按什么字符分割  chunk_size：分割最大长度  chunk_overlap：分割后重叠的长度
  text_splitter = CharacterTextSplitter(separator=",", chunk_size=5, chunk_overlap=2)
  # 字符串分割
  texts = text_splitter.split_text("hello,world,my,name,is,xiaodai")
  print(texts)
  # 文档分割 - 原数据不切，每个文档都有
  docs = text_splitter.split_documents([Document(metadata={'id': '1'}, page_content="hello,world,my,name,is,xiaodai"),
                                        Document(metadata={'name': '数字'}, page_content="1,2,3,4,5")])
  print(docs)
  # 将字符串切割成文档 - 元数据为空
  docs1 = text_splitter.create_documents(["hello,world,my,name,is,xiaodai", "1,2,3,4,5"])
  print(docs1)
  ~~~

- RecursiveCharacterTextSplitter（递归字符文本分割器）

  > 运行流程：
  >
  > - 首先尝试使用第一个分隔符（如 "\n\n"）分割文本
  > - 如果分割后的块仍然过大，则使用下一个分隔符继续分割
  > - 重复此过程，直到达到指定的 chunk_size 或用完所有分隔符

  ~~~python
  from langchain_text_splitters import RecursiveCharacterTextSplitter
  
  # 实例化分割器对象
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=20,  # 每个块最多 20 个字符
      chunk_overlap=6,  # 相邻分块会共享 6 个字符
      length_function=len,  # 用字符数来衡量长度
      separators=["\n\n", "\n", " ", ""]  # 会优先尝试按 \n\n 分段，如果太长，再按 \n → 空格 → 逐字符切分。
  )
  
  text = """
  人工智能正在快速发展，尤其是大语言模型的应用，正在改变人类的工作方式。
  它们可以帮助人们进行写作、代码生成、甚至是科研探索。
  相比之下，新能源的发展同样重要。
  电动车和太阳能正在逐渐替代传统能源，减少碳排放，对全球环境保护至关重要。
  """
  docs = text_splitter.split_text(text)
  print(docs)
  ~~~

- SemanticChunker（语义文档分割器）

  > 基于语义相似性分割文本

  ~~~python
  from langchain_experimental.text_splitter import SemanticChunker
  from langchain_ollama import OllamaEmbeddings
  
  embed = OllamaEmbeddings(model="bge-m3")
  
  text_splitter = SemanticChunker(
      embeddings=embed,
      # 断点阈值类型 -> 百分比
      breakpoint_threshold_type='percentile',
      # 更低的百分位会更“积极”地切分（数值越小 => 切得越多）
      # 把全部相邻句对的语义距离从小到大排序；取第 70% 位置的距离作为切分阈值
      # 距离 > 该阈值的句子对就切分
      breakpoint_threshold_amount=70.0,
      # 句子拆分正则为同时识别中/英文终结符
      sentence_split_regex=r'(?<=[。！？.!?])\s*',
      # 每个切分后片段（chunk）至少要有多少个字符
      min_chunk_size=10
  )
  
  text = """
  人工智能正在快速发展，尤其是大语言模型的应用，正在改变人类的工作方式。
  它们可以帮助人们进行写作、代码生成、甚至是科研探索。
  相比之下，新能源的发展同样重要。
  电动车和太阳能正在逐渐替代传统能源，减少碳排放，对全球环境保护至关重要。
  """
  
  docs = text_splitter.split_text(text)
  for i, d in enumerate(docs):
      print(f"------ Chunk {i + 1} ------")
      print(d.strip())
      print()
  ~~~

- MarkdownHeaderTextSplitter（Markdown文档切割器）

  > 适用于Markdown文档，按照标题进行拆分

  ~~~python
  from langchain_text_splitters import MarkdownHeaderTextSplitter
  
  # 给每个标题起别名，作为元数据的key
  headers_to_split_on = [
      ("#", "Header 1"),
      ("##", "Header 2"),
      ("###", "Header 3"),
  ]
  
  # 初始化分割器
  markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
  
  with open('./data/rag_tutorial.md', 'r', encoding='utf-8') as f:
      docs = markdown_splitter.split_text(f.read())
  print(docs)
  ~~~

##### 2.7.3 VectorStores（向量存储）

> 专门用来保存文本的向量嵌入 (embedding)，支持相似度检索
>
> LangChain支持的VectorStore，常见的如下：
>
> | VectorStore   | 描述                                                         |
> | ------------- | ------------------------------------------------------------ |
> | Chroma        | 一个开源嵌入式数据库                                         |
> | ElasticSearch | ElasticSearch                                                |
> | Milvus        | 用于存储、索引和管理由深度神经网络和其他机器学习（ML）模型产生的大量嵌入向量的数据库 |
> | Redis         | 基于 Redis 的检索器                                          |
> | FAISS         | Facebook AI 相似性搜索服务                                   |
> | Pinecone      | 一个具有广泛功能的向量数据库                                 |

~~~python
# 以Chroma为例
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma  # 向量库 单机运行
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings

# 1.加载文档
# 创建 TextLoader 对象
loader = TextLoader('./data/pku.txt', encoding='utf-8')
# 加载文档
docs = loader.load()
# print(f'docs-->{docs}')

# 2.将文档进行分块
# 创建 CharacterTextSplitter 对象
text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=200, chunk_overlap=30)
# 分块
split_docs = text_splitter.split_documents(docs)
print(f'split_docs-->{split_docs}')

# 3.将分割后的文档存储到向量数据库中
# 加载embedding模型
embedding = OllamaEmbeddings(model="bge-m3")
# 创建向量数据库，需要指定 存储的文档和向量模型名称以及持久化目录
chromadaDB = Chroma.from_documents(documents=split_docs,
                                   embedding=embedding,
                                   persist_directory='./data/chroma_db')

# 假如你的向量数据库已经存在，那么可以直接加载
# chromadaDB = Chroma(persist_directory='./chroma_db', embedding_function=embedding)

# 4.使用向量数据库进行查询
query = "1937年北京大学发生了什么？"
result = chromadaDB.similarity_search(query, k=2)
print(f'result-->{result}')
~~~

> vectordb.as_retriever() 和 vectordb.similarity_search()的区别：
>
> - 相同
>
>   - 核心技术：两者都基于向量相似度（如余弦相似度）从向量数据库中检索与查询最相关的文档。
>   - 底层技术：通常使用相同的嵌入模型和相似度计算方式（如 FAISS、Chroma、Pinecone 等）。
>
> - 不同
>
>   | 特性      | vectordb.as_retriever()                          | vectordb.similarity_search()             |
>   | --------- | ------------------------------------------------ | ---------------------------------------- |
>   | 返回类型  | 返回一个 Retriever 对象，用于 LangChain 链式调用 | 直接返回匹配的文档列表（List[Document]） |
>   | 使用场景  | 集成到 LangChain 的链（Chain）或代理（Agent）中  | 直接用于独立查询，无需链式调用           |
>   | 灵活性    | 支持通过参数配置检索策略（如 search_type="mmr"） | 需手动实现高级功能（如 MMR、过滤等）     |
>   | 输入/输出 | 作为组件接收链中的输入，输出标准化格式           | 直接接收查询字符串，返回原始文档列表     |
>   | 功能扩展  | 可结合其他检索方式（如混合检索、分级检索）       | 仅支持基础的相似度搜索                   |

##### 2.7.4 检索器

> 检索器是 LangChain 中负责信息检索的模块，通常与 向量存储（Vector Stores） 配合，通过嵌入模型（Embedding Models）将查询和文档转为向量，基于相似性进行检索：
>
> - 输入：接收用户查询（通常是文本）。
> - 处理：根据查询从数据源中检索相关内容。
> - 输出：返回一组相关文档或文本片段（通常是 Document 对象列表）。

> 常用检索器：
>
> | 检索器                           | 介绍                               |
> | :------------------------------- | :--------------------------------- |
> | Azure Cognitive Search Retriever | Amazon ACS检索服务                 |
> | ChatGPT Plugin Retriever         | ChatGPT检索插件                    |
> | Databerry                        | Databerry检索                      |
> | ElasticSearch BM25               | ElasticSearch检索器                |
> | Metal                            | Metal检索器                        |
> | Pinecone Hybrid Search           | Pinecone检索服务                   |
> | SVM Retriever                    | SVM检索器                          |
> | TF-IDF Retriever                 | TF-IDF检索器                       |
> | VectorStore Retriever            | VectorStore检索器                  |
> | Vespa retriever                  | 一个支持结构化文本和向量搜索的平台 |
> | Weaviate Hybrid Search           | 一个开源的向量搜索引擎             |
> | Wikipedia                        | 支持wikipedia内容检                |

- VectorStoreRetriever

  > as_retriever() 方法的 search_type 参数决定了向量检索的具体算法和行为

  ~~~python
  retriever = vector_store.as_retriever(
      search_type="similarity",  # 可选 "similarity"|"mmr"|"similarity_score_threshold"
      search_kwargs={
          "k": 5,  # 返回结果数量
          "score_threshold": 0.7,  # 仅当search_type="similarity_score_threshold"时有效,低于阈值的都丢弃。
          "filter": {"source": "重要文档.pdf"},  # 元数据过滤，只会检索满足条件的文档。
          "lambda_mult": 0.25  # 仅MMR搜索有效(控制多样性)：接近 0 则更强调和查询的相关性；接近 1 则更强调结果之间的差异性
      }
  )
  ~~~

  > 三种搜索类型对比：
  >
  > | 类型                       | 核心目标           | 结果数量  | 是否控制多样性 | 典型应用场景   |
  > | -------------------------- | ------------------ | --------- | -------------- | -------------- |
  > | similarity                 | 精确匹配           | 固定 k 个 | ❌              | 事实性问题回答 |
  > | mmr                        | 平衡相关性与多样性 | 固定 k 个 | ✅              | 生成综合性报告 |
  > | similarity_score_threshold | 质量过滤           | 动态数量  | ❌              | 高精度筛选     |


##### 2.7.5 常用检索器

- TFIDFRetriever（稀疏检索）

  > 基于 TF-IDF（词频-逆文档频率）的检索器
  >
  > TF：词频，某个词在**当前文档**里出现的次数
  >
  > IDF：逆文档频率，衡量这个词在**全部文档集合**里有多稀有，出现越少IDF越大
  >
  > TF‑IDF：TF*IDF

- BM25Retriever（稀疏检索）

  > BM25 是对 TF-IDF 的改进版本，在 TF-IDF 基础上做了归一化（防止长文档优势）与饱和控制（防止词频无限增长）

#### 2.8 Agent

> **Agent其实就是基于大模型的语义理解和推理能力，让大模型拥有解决复杂问题时的任务规划能力，并调用外部工具来执行各种任务，利用向量数据库保留“记忆”的一个智能体**
>
> Agent = 大模型 + 任务规划（Planning） + 使用外部工具执行任务（Tools&Action） + 记忆（Memory）

![PixPin_2026-08-27_16-51-38](.\assets\PixPin_2026-08-27_16-51-38.png)

>  langchain实现Agent(部分)：
>
> - **Zero-shot ReAct Description**：
>   - 基于 **ReAct 框架**（推理 + 行动），仅依赖工具的 **描述** 来决定调用哪个工具
>   - 特点：无需额外示例（zero-shot），但 **不具备记忆能力**，每次推理都独立进行
> - **Structured Chat Zero-shot ReAct Description**
>   - 基于 **ReAct 框架**，但可以处理 **结构化输入**，即支持带多个参数的工具（类似函数调用）
>   - 不仅能根据描述选择工具，还能正确组织并传递复杂参数
> - **Conversational ReAct Description**
>   - 在 ReAct 框架基础上，增强了 **对话记忆能力**
>   - 能根据上下文对话历史来做出工具选择和回应，更适合持续性对话场景
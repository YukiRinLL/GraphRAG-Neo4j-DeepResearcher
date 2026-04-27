# GraphRAG-Neo4j-DeepResearcher

<p align="center">
<img src="https://img.shields.io/badge/python-%E2%89%A53.10-5be.svg">
<a href='https://ms-agent.readthedocs.io/zh-cn/latest/'>
    <img src='https://readthedocs.org/projects/ms-agent/badge/?version=latest' alt='Documentation Status' />
</a>
<a href="https://github.com/modelscope/ms-agent/actions?query=branch%3Amaster+workflow%3Acitest++"><img src="https://img.shields.io/github/actions/workflow/status/modelscope/ms-agent/citest.yaml?branch=master&logo=github&label=CI"></a>
<a href="https://github.com/modelscope/ms-agent/blob/main/LICENSE"><img src="https://img.shields.io/github/license/modelscope/modelscope-agent"></a>
<a href="https://github.com/modelscope/ms-agent/pulls"><img src="https://img.shields.io/badge/PR-welcome-55EB99.svg"></a>
<a href="https://pypi.org/project/ms-agent/"><img src="https://badge.fury.io/py/ms-agent.svg"></a>
<a href="https://pepy.tech/project/ms-agent"><img src="https://static.pepy.tech/badge/ms-agent"></a>
</p>



[**README**](README.md)

## 简介

GraphRAG-Neo4j-DeepResearcher 是一个增强型深度研究智能体框架，集成了 MS-Agent、GraphRAG 结构化知识提取和 Neo4j 图数据库，用于长文本理解、多文档推理和持久化知识图谱存储。

### 新增核心特性

- **GraphRAG 集成**：结合图数据库与检索增强生成，提升知识表示和推理能力
- **Neo4j 知识图谱**：持久化存储实体、关系和文档嵌入
- **高级实体提取**：多模式实体识别，包含质量过滤和归一化
- **关系提取**：基于语义的实体间关系识别
- **嵌入集成**：通过 SiliconFlow API 使用 BAAI/bge-m3 模型进行文本嵌入
- **可配置知识图谱工具**：易于与现有 MS-Agent 工作流集成
- **增强配置文件**：所有配置选项的详细中文注释

## 项目结构

```
├── ms_agent/
│   └── tools/
│       └── knowledge_graph/     # 知识图谱工具实现
│           ├── __init__.py
│           ├── config.py         # 知识图谱配置
│           ├── knowledge_graph_builder.py  # 实体和关系提取
│           ├── knowledge_graph_retriever.py  # 知识图谱检索
│           ├── knowledge_graph_tool.py  # 工具接口
│           └── neo4j_connection.py  # Neo4j 数据库连接
├── projects/
│   └── deep_research/
│       └── v2/
│           ├── researcher.yaml   # 研究者代理配置
│           ├── reporter.yaml     # 报告生成代理配置
│           └── searcher.yaml     # 搜索和证据收集代理配置
├── .env                         # 环境变量
├── run_benchmark.ps1            # 基准测试脚本
└── test_kg_build.py             # 知识图谱构建测试
```

## 配置

### 环境变量 (.env)

项目使用以下环境变量进行配置：

```env
# 搜索 API 密钥
EXA_API_KEY=<EXA_API_KEY>
SERPAPI_API_KEY=<SERPAPI_API_KEY>

# LLM 配置
OPENAI_API_KEY=<OPENAI_API_KEY>
OPENAI_BASE_URL=<OPENAI_BASE_URL>

# Neo4j 配置
NEO4J_URI=<NEO4J_URI>
NEO4J_USERNAME=<NEO4J_USERNAME>
NEO4J_PASSWORD=<NEO4J_PASSWORD>

# 嵌入配置
EMBEDDING_PROVIDER=<EMBEDDING_PROVIDER>
EMBEDDING_MODEL=<EMBEDDING_MODEL>
SILICONFLOW_API_KEY=<SILICONFLOW_API_KEY>
SILICONFLOW_API_URL=<SILICONFLOW_API_URL>
```

### 代理配置文件

项目包含三个主要配置文件，带有详细的中文注释：

1. **researcher.yaml**：主要协调器代理配置
2. **reporter.yaml**：报告生成代理配置  
3. **searcher.yaml**：搜索和证据收集代理配置

关键配置选项：

- **LLM 设置**：服务提供商、模型名称、API 密钥
- **工具配置**：文件系统、代码执行器、证据存储
- **知识图谱集成**：通过插件系统启用
- **生成设置**：流式输出、前缀缓存、思考模式

## 使用方法

### 运行基准测试

```powershell
# 运行深度研究基准测试
.projects\deep_research\v2\run_benchmark.ps1
```

### 测试知识图谱构建

```powershell
# 测试知识图谱构建
python test_kg_build.py
```

## 知识图谱特性

### 实体提取
- **多模式提取**：大写单词、技术术语、缩写词、多词短语
- **实体归一化**：跨文档的一致实体命名
- **质量过滤**：移除低质量和元数据实体
- **实体类型推断**：自动分类实体类型

### 关系提取
- **语义关系识别**：基于上下文指示器
- **关系质量评分**：过滤无关关系
- **关系去重**：避免冗余连接

### Neo4j 集成
- **持久化存储**：实体、关系和嵌入
- **向量索引**：用于基于相似度的检索
- **Cypher 查询**：高级图遍历和分析

## 主要修改

1. **添加知识图谱工具** (`ms_agent/tools/knowledge_graph/`)
2. **增强实体提取**，支持多模式和质量过滤
3. **集成 Neo4j** 用于图存储和检索
4. **添加嵌入支持**，通过 SiliconFlow API
5. **更新配置文件**，添加中文注释
6. **修复编码问题**，添加 Latin-1 作为 fallback
7. **添加基准测试脚本** (`run_benchmark.ps1`)
8. **改进错误处理**，增强 API 连接稳定性

## 许可证

本项目基于 [Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE) 许可证。

---
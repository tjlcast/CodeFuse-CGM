根据代码和导入的模块分析，代码图中的节点和边类型如下：

## 节点类型 (Node Types)

从导入的 [NodeType](file://c:\Users\phx10\code\CodeFuse-CGM\reranker\codegraph_parser\java\codegraph_java_local.py#L6-L16) 可以看出，节点类型包括：

- [NodeType.REPO](file://c:\Users\phx10\code\CodeFuse-CGM\reranker\codegraph_parser\java\codegraph_java_local.py#L7-L7): 仓库节点 (Repository)
- [NodeType.PACKAGE](file://c:\Users\phx10\code\CodeFuse-CGM\reranker\codegraph_parser\java\codegraph_java_local.py#L8-L8): 包节点 (Package)
- [NodeType.FILE](file://c:\Users\phx10\code\CodeFuse-CGM\reranker\codegraph_parser\java\codegraph_java_local.py#L9-L9): 文件节点 (File)

此外，根据 [codegraph_parser.python.codegraph_python_local](file://c:\Users\phx10\code\CodeFuse-CGM\retriever\codegraph_parser\python\codegraph_python_local.py) 的导入，应该还包含Python特定的节点类型：

- 类节点 (Class)
- 函数节点 (Function)
- 方法节点 (Method)
- 变量节点 (Variable)
- 模块节点 (Module)

从代码中可以看出，处理时会跳过 [NodeType.REPO](file://c:\Users\phx10\code\CodeFuse-CGM\reranker\codegraph_parser\java\codegraph_java_local.py#L7-L7) 和 [NodeType.PACKAGE](file://c:\Users\phx10\code\CodeFuse-CGM\reranker\codegraph_parser\java\codegraph_java_local.py#L8-L8) 类型的节点：
```python
if node_type in [NodeType.REPO, NodeType.PACKAGE]:
    continue
```

## 边类型 (Edge Types)

从导入的 [EdgeType](file://c:\Users\phx10\code\CodeFuse-CGM\reranker\codegraph_parser\java\codegraph_java_local.py#L19-L23) 可以看出，边类型包括：

- [EdgeType.CONTAINS](file://c:\Users\phx10\code\CodeFuse-CGM\reranker\codegraph_parser\java\codegraph_java_local.py#L20-L20): 包含关系 (Contains)
  - 在 [subgraph.py](file://c:\Users\phx10\code\CodeFuse-CGM\retriever\subgraph.py) 中有使用，用于构建从节点到仓库的路径：
  ```python
  if graph_nx[pre_node][node][0]['type'] == EdgeType.CONTAINS:
  ```

根据代码图的一般结构，可能还包含其他类型的边：

- 继承关系 (Inheritance)
- 调用关系 (Call)
- 引用关系 (Reference)
- 依赖关系 (Dependency)

## 代码图结构示例

典型的代码图层次结构如下：
```
REPO (仓库)
  ├── CONTAINS
  PACKAGE (包)
    ├── CONTAINS
    FILE (文件)
      ├── CONTAINS
      CLASS (类)
        ├── CONTAINS
        METHOD (方法)
          ├── CONTAINS
          VARIABLE (变量)
```

## 实际处理中的体现

在 [generate_code_content.py](file://c:\Users\phx10\code\CodeFuse-CGM\preprocess_embedding\generate_code_content.py) 中：
- 获取所有节点：`nodes = graph.get_nodes()`
- 提取节点内容：`content = node.get_content()`
- 根据节点ID存储代码和文档内容

在 [subgraph.py](file://c:\Users\phx10\code\CodeFuse-CGM\retriever\subgraph.py) 中：
- 使用 [EdgeType.CONTAINS](file://c:\Users\phx10\code\CodeFuse-CGM\reranker\codegraph_parser\java\codegraph_java_local.py#L20-L20) 来构建从节点到仓库的路径
- 保证子图的连通性

这种结构使得代码图能够表达完整的代码结构和依赖关系，支持精确的代码检索和理解。
from pathlib import Path
# 中文导读：动态工具注册。模型能否使用新工具，取决于模块是否满足这份接口契约。
import importlib

def load_all_tools(logging=print):
    # 扫描同目录 .py 文件；每个模块必须提供 tool_info（名称/描述/schema）和
    # tool_function（真正执行动作）。返回的 info 提供给模型，function 留在 Python 执行。
    # 因此进化代理只要新增合规工具文件，下次启动即可发现，无需硬编码注册清单。
    tools_dir = Path(__file__).parent
    tools = []
    
    # Get all Python files in the tools directory (excluding __init__.py)
    tool_files = [f for f in tools_dir.glob("*.py") if f.stem != "__init__"]
    
    for tool_file in tool_files:
        # Import the module
        module_name = f"tools.{tool_file.stem}"
        try:
            module = importlib.import_module(module_name)
            
            # Check if module has required functions
            if hasattr(module, 'tool_info') and hasattr(module, 'tool_function'):
                tools.append({
                    'info': module.tool_info(),
                    'function': module.tool_function,
                    'name': tool_file.stem
                })
            else:
                raise Exception(f"Tool module {module_name} does not have required functions.")
        except Exception as e:
            # Log the error and raise it
            logging(f"导入模块失败：{module_name}: {e}")
            raise e
    
    return tools

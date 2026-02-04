import os
from typing import List, Dict, Any, Optional


class FileOperator:
    def read_file(self, filepath: str) -> Dict[str, Any]:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            return {
                "status": "success",
                "content": content,
                "path": filepath,
                "size": os.path.getsize(filepath)
            }
        except FileNotFoundError:
            return {"status": "error", "message": "File not found"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def write_file(self, filepath: str, content: str) -> Dict[str, Any]:
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return {
                "status": "success",
                "path": filepath,
                "size": len(content)
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def append_file(self, filepath: str, content: str) -> Dict[str, Any]:
        try:
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(content)
            return {
                "status": "success",
                "path": filepath,
                "size": os.path.getsize(filepath)
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def list_directory(self, path: str = ".") -> Dict[str, Any]:
        try:
            items = os.listdir(path)
            files = []
            dirs = []
            for item in items:
                full_path = os.path.join(path, item)
                if os.path.isfile(full_path):
                    files.append(item)
                elif os.path.isdir(full_path):
                    dirs.append(item)
            return {
                "status": "success",
                "path": path,
                "files": files,
                "directories": dirs
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def create_directory(self, path: str) -> Dict[str, Any]:
        try:
            os.makedirs(path, exist_ok=True)
            return {"status": "success", "path": path}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def delete_file(self, filepath: str) -> Dict[str, Any]:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                return {"status": "success", "path": filepath}
            else:
                return {"status": "error", "message": "File not found"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def search_files(self, directory: str, pattern: str) -> Dict[str, Any]:
        try:
            matches = []
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if pattern in file:
                        matches.append(os.path.join(root, file))
            return {
                "status": "success",
                "matches": matches,
                "count": len(matches)
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_file_info(self, filepath: str) -> Dict[str, Any]:
        try:
            if not os.path.exists(filepath):
                return {"status": "error", "message": "File not found"}
            
            stat = os.stat(filepath)
            return {
                "status": "success",
                "path": filepath,
                "size": stat.st_size,
                "created": stat.st_ctime,
                "modified": stat.st_mtime,
                "is_file": os.path.isfile(filepath),
                "is_directory": os.path.isdir(filepath)
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

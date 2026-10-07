@echo off
rem ============================================================
rem  日志故障诊断 Agent - Git 仓库初始化脚本
rem  用法: 双击运行, 或在 cmd / PowerShell 中执行 init_git.bat
rem  效果: git init -> 添加代码文件(自动排除大数据集/密钥) -> 首次 commit
rem  说明: 本脚本不会推送远程仓库; 推送 GitHub 步骤见 README.md
rem ============================================================
chcp 65001 >nul
cd /d "%~dp0"

echo [1/4] 检查并创建 db 目录占位文件...
if not exist "db" mkdir "db"
if not exist "db\.gitkeep" type nul > "db\.gitkeep"

echo [2/4] git init ...
if exist ".git" (
    echo    已存在 .git 仓库, 跳过 init
) else (
    git init
)

echo [3/4] 添加文件(.gitignore 已排除 .env / 大数据集 / venv / chroma / 日志)...
git add .

echo [4/4] 首次提交...
git rev-parse --verify HEAD >nul 2>&1
if %errorlevel%==0 (
    echo    仓库已有提交, 跳过 commit(请手动 git add . && git commit)
) else (
    git commit -m "feat: 日志故障诊断 Agent(FastAPI + LangGraph + RAG + Chroma + 智谱GLM)"
)

echo.
echo ============================================================
echo  Git 初始化完成。已忽略: .env(密钥)、HDFS.log、anomaly_label.csv、
echo  rag/data/chroma(向量库)、.venv、日志缓存、SQLite 工单库。
echo  推送 GitHub 的完整步骤见 README.md「GitHub 提交与推送」章节。
echo ============================================================
pause

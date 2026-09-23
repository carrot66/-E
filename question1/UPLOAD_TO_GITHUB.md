# 上传到 carrot66/-E 的分支

本地整理目录：`E:\DeskTop\数学建模\E题\question1_github`。

下面把它作为仓库中的 `question1/` 文件夹提交，不改动你的原始 `question1` 工作目录。整理过程没有登录GitHub或执行远程推送。当前网络无法查询该仓库，所以不能确认仓库是否公开、已有分支名称或你的写权限。

## 方法一：Git命令上传（推荐）

在Windows PowerShell逐段运行。先克隆现有仓库，保留原仓库历史：

```powershell
Set-Location -LiteralPath 'E:\DeskTop\数学建模\E题'
git clone https://github.com/carrot66/-E.git github_E_upload
Set-Location -LiteralPath 'E:\DeskTop\数学建模\E题\github_E_upload'
git branch -r
```

如果 `github_E_upload` 已存在，请直接进入已有克隆目录，执行 `git status` 确认没有待处理修改，再执行 `git fetch origin`；不要重复克隆或删除已有工作。

**新建一个分支上传：**以下名称是建议的新分支名，不代表远程已经存在。

```powershell
git switch -c question1-feature-extraction
```

**如果你要上传到已有分支，则不要执行上面的新建分支命令。**改为执行 `git switch --track origin/你的实际分支名`；若本地已有该分支，使用 `git switch 你的实际分支名`。用前面的 `git branch -r` 确认名称，不要原样输入“你的实际分支名”。

复制文件并检查变更：

```powershell
$sourceFolder = 'E:\DeskTop\数学建模\E题\question1_github'
$targetFolder = 'E:\DeskTop\数学建模\E题\github_E_upload\question1'
New-Item -ItemType Directory -Path $targetFolder -Force | Out-Null
Get-ChildItem -LiteralPath $sourceFolder -Force | Copy-Item -Destination $targetFolder -Recurse -Force
git status --short
git add -- question1
git diff --cached --stat
```

复制会更新同名文件，保留已有分支中的额外文件；如果分支原来已有 `question1`，请检查差异，尤其是原来已经被Git跟踪的大文件（`.gitignore` 不会自动取消跟踪旧文件）。

确认暂存内容后提交并推送当前分支：

```powershell
git commit -m "Add question1 multimodal feature pipeline and results"
git push -u origin HEAD
```

若Git提示未设置作者，仅在这个仓库设置你的真实身份后再提交：

```powershell
git config user.name "你的GitHub用户名"
git config user.email "你的GitHub提交邮箱"
```

推送时按Git Credential Manager提示登录有仓库写权限的GitHub账号。完成后访问 https://github.com/carrot66/-E/branches ，选择刚上传的分支，进入 `question1` 即可查看。

如果推送被拒绝，不要使用强制推送；先根据错误检查分支保护、登录权限或远程新增提交。

## 方法二：GitHub网页上传

在仓库中选择或新建目标分支，打开 `Add file` → `Upload files`。先在本地单独建一个待上传的 `question1` 文件夹，把 `question1_github` 的内容复制进去，再拖入网页；不要把原始含模型和虚拟环境的 `question1` 拖进去。文件较多时建议使用上面的Git方式，避免网页批量上传限制。

## 音视频和模型如何保存

当前文件夹包含全部100条样本特征，但不包含完整音视频输出和模型。原 `question1/outputs_v3` 与 `question1/models` 保持原样，可继续用于分析和服务器运行。需要让他人完整复现时，可把原有模型压缩包及完整结果压缩包作为独立附件或GitHub Release附件分享，并提供来源与校验值；不要直接加入普通Git提交。

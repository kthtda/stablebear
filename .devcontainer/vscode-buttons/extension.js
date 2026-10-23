// A Stablebear panel in the activity bar whose actions run development
// commands as tasks or open built files in the browser. The bear above the
// editor opens the panel.
// To add an action, add an entry to ACTIONS and its command to "commands" in
// package.json.
// The devcontainers install it on attach with install.sh. After a change,
// raise "version" in package.json (VS Code caches an extension by version),
// run install.sh, then run "Developer: Reload Window".
const fs = require("fs");
const path = require("path");
const vscode = require("vscode");

const ACTIONS = [
  {
    command: "stablebear.buildDocs",
    label: "Build docs",
    cwd: "docs",
    shell: "make html",
    // Sphinx fails with "unsupported locale setting" under the default locale.
    env: { LC_ALL: "C.UTF-8", LANG: "C.UTF-8" },
  },
  {
    command: "stablebear.openDocs",
    label: "Open docs",
    open: "docs/_build/html/index.html",
  },
];

// The browser runs on the host, so it needs the host's path to the
// repository, which the devcontainer passes as SB_HOST_WORKSPACE.
function openInBrowser(action) {
  const folder = vscode.workspace.workspaceFolders?.[0];
  if (!folder) {
    return;
  }
  if (!fs.existsSync(path.join(folder.uri.fsPath, action.open))) {
    vscode.window.showWarningMessage(`${action.open} does not exist yet. Build the docs first.`);
    return;
  }
  const root = process.env.SB_HOST_WORKSPACE || folder.uri.fsPath;
  vscode.env.openExternal(vscode.Uri.file(path.posix.join(root, action.open)));
}

function run(action) {
  const folder = vscode.workspace.workspaceFolders?.[0];
  if (!folder) {
    return;
  }
  const execution = new vscode.ShellExecution(action.shell, {
    cwd: path.join(folder.uri.fsPath, action.cwd),
    env: action.env,
  });
  const task = new vscode.Task({ type: "stablebear" }, folder, action.label, "stablebear", execution, []);
  task.presentationOptions = { reveal: vscode.TaskRevealKind.Always, clear: true };
  vscode.tasks.executeTask(task);
}

function executionOf(action) {
  return vscode.tasks.taskExecutions.find((e) =>
    e.task.source === "stablebear" && e.task.name === action.label);
}

// The button of a running action stops it instead.
function toggle(action) {
  if (action.open) {
    openInBrowser(action);
    return;
  }
  const execution = executionOf(action);
  if (execution) {
    execution.terminate();
  } else {
    run(action);
  }
}

const PLAY = '<path class="play" d="M4 2.5v11l9-5.5z"/>';
const STOP = '<rect class="stop" x="3.5" y="3.5" width="9" height="9" rx="1"/>';
const OPEN = '<path class="open" d="M9 2h5v5h-1.5V4.6L7.6 9.5 6.5 8.4l4.9-4.9H9zM3 4h4v1.5H4.5v6h6V9H12v4H3z"/>';

function button(action, running) {
  if (action.open) {
    return { title: `${action.label} in the browser`, icon: OPEN };
  }
  return {
    title: `${running ? "Stop" : "Run"} ${action.label}`,
    icon: running ? STOP : PLAY,
  };
}

// Tree view inline actions only show on hover, so the panel is a webview with
// an always-visible play button per action; clicking the row does nothing.
function actionsHtml(webview) {
  const nonce = Math.random().toString(36).slice(2);
  const rows = ACTIONS.map((action, i) => {
    const { title, icon } = button(action, !!executionOf(action));
    return `
    <div class="row" title="${action.open ?? `${action.shell} in ${action.cwd}/`}">
      <span>${action.label}</span>
      <button data-index="${i}" title="${title}">
        <svg viewBox="0 0 16 16" width="16" height="16">${icon}</svg>
      </button>
    </div>`;
  }).join("");
  return `<!DOCTYPE html>
<html>
<head>
<meta http-equiv="Content-Security-Policy"
  content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';">
<style>
  body { padding: 4px 0; }
  .row { display: flex; align-items: center; justify-content: space-between; padding: 2px 8px 2px 20px; }
  .row:hover { background: var(--vscode-list-hoverBackground); }
  button { display: flex; padding: 2px; border: none; border-radius: 4px; background: none; cursor: pointer; }
  button:hover { background: var(--vscode-toolbar-hoverBackground); }
  button:focus-visible { outline: 1px solid var(--vscode-focusBorder); }
  .play { fill: var(--vscode-debugIcon-startForeground, var(--vscode-icon-foreground)); }
  .stop { fill: var(--vscode-debugIcon-stopForeground, var(--vscode-icon-foreground)); }
  .open { fill: var(--vscode-icon-foreground); }
</style>
</head>
<body>${rows}
<script nonce="${nonce}">
  const vscode = acquireVsCodeApi();
  for (const button of document.querySelectorAll("button")) {
    button.addEventListener("click", () => vscode.postMessage(Number(button.dataset.index)));
  }
  window.addEventListener("message", ({ data }) => {
    data.forEach(({ title, icon }, i) => {
      const button = document.querySelector(\`button[data-index="\${i}"]\`);
      button.title = title;
      button.firstElementChild.innerHTML = icon;
    });
  });
</script>
</body>
</html>`;
}

let actionsWebview;

function refresh() {
  actionsWebview?.postMessage(ACTIONS.map((action) => button(action, !!executionOf(action))));
}

const actionsView = {
  resolveWebviewView: (view) => {
    actionsWebview = view.webview;
    view.webview.options = { enableScripts: true };
    view.webview.html = actionsHtml(view.webview);
    view.webview.onDidReceiveMessage((index) => toggle(ACTIONS[index]));
    view.onDidDispose(() => {
      actionsWebview = undefined;
    });
  },
};

function activate(context) {
  for (const action of ACTIONS) {
    context.subscriptions.push(vscode.commands.registerCommand(action.command, () => toggle(action)));
  }
  context.subscriptions.push(vscode.tasks.onDidStartTask(refresh), vscode.tasks.onDidEndTask(refresh));
  context.subscriptions.push(vscode.window.registerWebviewViewProvider("stablebear.actions", actionsView));
  context.subscriptions.push(vscode.commands.registerCommand("stablebear.openPanel", () =>
    vscode.commands.executeCommand("workbench.view.extension.stablebear")));
}

module.exports = { activate };

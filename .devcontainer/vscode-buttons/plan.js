// The Plan view: the items of an implementation plan in plans/, as written
// by plans/PLANNING.md, in a tree of phases, items, and subitems, with the
// progress over the items at the top. The view's title bar selects the plan.
const fs = require("fs");
const path = require("path");
const vscode = require("vscode");

const PLAN_GLOB = "plans/*-plan.md";
const SELECTED = "stablebear.plan";

// "- [x] **B2b.** [3] (9629137bd, d37757337) text", with the points and the
// commits optional.
const ITEM = /^(\s*)- \[([ xX])\] \*\*([^*]+?)\.?\*\*\s*(?:\[(\d+)\]\s*)?(?:\(([0-9a-f]{7,}(?:,\s*[0-9a-f]{7,})*)\)\s*)?(.*)$/;
const PHASE = /^## (.*)$/;

// The phases of a plan, each with its items, each with its subitems. Lines
// indented under an item continue its text.
function parsePlan(text) {
  const phases = [];
  let item;
  let parent;
  text.split("\n").forEach((line, index) => {
    const phase = PHASE.exec(line);
    if (phase) {
      phases.push({ label: phase[1], line: index, children: [] });
      item = parent = undefined;
      return;
    }
    const match = ITEM.exec(line);
    if (match && phases.length) {
      item = {
        id: match[3],
        done: match[2] !== " ",
        points: Number(match[4] ?? 0),
        commits: match[5] ? match[5].split(/,\s*/) : [],
        text: match[6],
        line: index,
        children: [],
      };
      if (match[1].length && parent) {
        parent.children.push(item);
      } else {
        phases.at(-1).children.push(item);
        parent = item;
      }
      return;
    }
    if (item && /^\s+\S/.test(line)) {
      item.text += " " + line.trim();
    } else if (line.trim()) {
      item = undefined;
    }
  });
  return phases;
}

// Items without subitems, and subitems: the ones that carry points.
function workItems(phases) {
  return phases.flatMap((phase) => phase.children.flatMap((item) => item.children.length ? item.children : [item]));
}

function progress(phases) {
  const items = workItems(phases);
  const done = items.filter((item) => item.done);
  const points = items.reduce((sum, item) => sum + item.points, 0);
  const donePoints = done.reduce((sum, item) => sum + item.points, 0);
  const percent = points ? Math.round((100 * donePoints) / points) : 0;
  return { done: done.length, total: items.length, donePoints, points, percent };
}

function progressText(phases) {
  const p = progress(phases);
  return `${p.done} of ${p.total} items, ${p.donePoints} of ${p.points} points (${p.percent}%)`;
}

function plain(text) {
  return text.replace(/[`*]/g, "");
}

// The text up to its tests or notes.
function summary(text) {
  return plain(text.split(/\s(?:Tests|Note):/)[0]);
}

// The size of each item with points as a badge of its points on the right,
// as git marks changed files, so that the icon on the left keeps showing
// whether the item is done. Each such item has a resourceUri of this scheme
// whose query holds its points.
const ITEM_SCHEME = "stablebear-plan-item";

class SizeDecorations {
  provideFileDecoration(uri) {
    const points = Number(uri.query);
    if (uri.scheme !== ITEM_SCHEME || !points) {
      return undefined;
    }
    return new vscode.FileDecoration(String(points), `${points} point${points === 1 ? "" : "s"}`);
  }
}

function isDone(node) {
  return node.children.length ? node.children.every(isDone) : node.done;
}

const DONE = new vscode.ThemeIcon("pass-filled", new vscode.ThemeColor("testing.iconPassed"));
const TODO = new vscode.ThemeIcon("circle-large-outline");

class PlanProvider {
  constructor(context) {
    this.context = context;
    this.changed = new vscode.EventEmitter();
    this.onDidChangeTreeData = this.changed.event;
    this.file = undefined;
    this.phases = [];
  }

  planFiles() {
    const folder = vscode.workspace.workspaceFolders?.[0];
    if (!folder) {
      return [];
    }
    const dir = path.join(folder.uri.fsPath, "plans");
    let names = [];
    try {
      names = fs.readdirSync(dir).filter((name) => name.endsWith("-plan.md"));
    } catch {
      return [];
    }
    // Most recently changed first.
    return names.map((name) => path.join(dir, name))
      .sort((a, b) => fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs);
  }

  // The selected plan if it still exists, else the most recently changed.
  load() {
    const files = this.planFiles();
    const selected = this.context.workspaceState.get(SELECTED);
    this.file = files.includes(selected) ? selected : files[0];
    this.phases = this.file ? parsePlan(fs.readFileSync(this.file, "utf8")) : [];
  }

  refresh() {
    this.load();
    if (this.view) {
      this.view.description = this.file ? path.basename(this.file, ".md") : undefined;
      this.view.message = this.file ? progressText(this.phases) : "No plans/*-plan.md files";
    }
    this.changed.fire();
  }

  async select() {
    const picks = this.planFiles().map((file) => ({
      label: path.basename(file, ".md"),
      description: progressText(parsePlan(fs.readFileSync(file, "utf8"))),
      file,
    }));
    const pick = await vscode.window.showQuickPick(picks, { placeHolder: "Select a plan" });
    if (pick) {
      await this.context.workspaceState.update(SELECTED, pick.file);
      this.refresh();
    }
  }

  getChildren(node) {
    return node ? node.children : this.phases;
  }

  getTreeItem(node) {
    const isPhase = node.id === undefined;
    const done = isDone(node);
    // Expand what is still to do.
    const state = !node.children.length ? vscode.TreeItemCollapsibleState.None
      : done ? vscode.TreeItemCollapsibleState.Collapsed : vscode.TreeItemCollapsibleState.Expanded;
    const item = new vscode.TreeItem(isPhase ? plain(node.label) : `${node.id}  ${summary(node.text)}`, state);
    item.iconPath = done ? DONE : TODO;
    if (!isPhase && node.points) {
      item.resourceUri = vscode.Uri.from({
        scheme: ITEM_SCHEME,
        path: `/${path.basename(this.file)}/${node.id}`,
        query: String(node.points),
      });
    }
    if (isPhase) {
      const items = workItems([node]);
      item.description = `${items.filter((i) => i.done).length}/${items.length}`;
    } else {
      item.description = node.commits.join(", ");
      const points = node.points ? ` [${node.points}]` : "";
      item.tooltip = new vscode.MarkdownString(`**${node.id}.**${points} ${node.text}`);
    }
    item.command = {
      command: "vscode.open",
      title: "Open in plan",
      arguments: [vscode.Uri.file(this.file), { selection: new vscode.Range(node.line, 0, node.line, 0) }],
    };
    return item;
  }
}

function activatePlan(context) {
  const provider = new PlanProvider(context);
  provider.view = vscode.window.createTreeView("stablebear.plan", { treeDataProvider: provider });
  provider.refresh();
  context.subscriptions.push(provider.view);
  context.subscriptions.push(vscode.commands.registerCommand("stablebear.selectPlan", () => provider.select()));
  context.subscriptions.push(vscode.window.registerFileDecorationProvider(new SizeDecorations()));
  const folder = vscode.workspace.workspaceFolders?.[0];
  if (folder) {
    const watcher = vscode.workspace.createFileSystemWatcher(new vscode.RelativePattern(folder, PLAN_GLOB));
    watcher.onDidChange(() => provider.refresh());
    watcher.onDidCreate(() => provider.refresh());
    watcher.onDidDelete(() => provider.refresh());
    context.subscriptions.push(watcher);
  }
}

module.exports = { activatePlan, parsePlan, progress };

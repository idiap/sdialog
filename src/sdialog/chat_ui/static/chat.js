/*
 * SDialog chat GUI. Conversations are kept in the browser's localStorage and sent in full with each message,
 * so the server keeps nothing.
 *
 * SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
 * SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
 * SPDX-License-Identifier: MIT
 */
(function () {
  "use strict";

  var config = JSON.parse(document.getElementById("sdialog-chat-config").textContent);
  var params = new URLSearchParams(location.search);
  var embedded = params.has("embed") ? params.get("embed") !== "0" : isInFrame();
  var inWidget = params.get("widget") === "1";
  var STORAGE_VERSION = 1;

  var el = {};
  ["app", "sidebar", "scrim", "conversation-list", "new-chat", "new-chat-side", "open-sidebar", "close-sidebar",
   "close-widget", "title", "subtitle", "header-avatar", "messages", "composer", "input", "send"
  ].forEach(function (id) { el[id] = document.getElementById(id); });

  // In-memory state; only `data` is saved
  var data = load();
  var pending = {};  // conversation id -> true while waiting for the agent's reply
  var errors = {};  // conversation id -> error message of the last failed request
  var draftModel = config.agents.length ? config.agents[0].id : null;  // agent of the next new conversation
  var shown = { id: null, count: 0 };  // conversation and number of messages last rendered (to animate new ones)

  function isInFrame() {
    try { return window.self !== window.top; } catch (e) { return true; }
  }

  /* Storage */

  var memoryStore = null;  // used when localStorage is unavailable (e.g. blocked in third-party iframes)

  function emptyData() {
    return { version: STORAGE_VERSION, current: null, conversations: {} };
  }

  function load() {
    var raw = null;
    try { raw = window.localStorage.getItem(config.storageKey); } catch (e) { raw = memoryStore; }
    try {
      var parsed = raw ? JSON.parse(raw) : null;
      if (parsed && parsed.version === STORAGE_VERSION && parsed.conversations) return parsed;
    } catch (e) { /* corrupted: start over */ }
    return emptyData();
  }

  function save() {
    var raw = JSON.stringify(data);
    try {
      window.localStorage.setItem(config.storageKey, raw);
    } catch (e) {
      memoryStore = raw;
    }
  }

  /* Conversations */

  function newId() {
    return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
  }

  function agentName(model) {
    for (var i = 0; i < config.agents.length; i++) {
      if (config.agents[i].id === model) return config.agents[i].name;
    }
    return model || "Agent";
  }

  function current() {
    return data.current ? data.conversations[data.current] || null : null;
  }

  function sortedConversations() {
    return Object.keys(data.conversations).map(function (id) { return data.conversations[id]; })
      .sort(function (a, b) { return b.updated - a.updated; });
  }

  function createConversation(firstMessage) {
    if (!config.multipleConversations) data.conversations = {};
    var now = Date.now();
    var conv = {
      id: newId(), title: makeTitle(firstMessage), model: draftModel,
      messages: [], created: now, updated: now
    };
    data.conversations[conv.id] = conv;
    data.current = conv.id;
    return conv;
  }

  function makeTitle(text) {
    text = text.replace(/\s+/g, " ").trim();
    return text.length > 48 ? text.slice(0, 47).trim() + "…" : text;
  }

  function startNewChat() {
    var conv = current();
    if (!config.multipleConversations && conv && conv.messages.length) {
      if (!window.confirm("Start a new chat? The current conversation will be deleted.")) return;
      data.conversations = {};
    }
    if (conv) draftModel = conv.model;
    data.current = null;
    save();
    closeDrawer();
    render();
    el.input.focus();
  }

  function openConversation(id) {
    data.current = id;
    save();
    closeDrawer();
    render();
  }

  function deleteConversation(id) {
    var conv = data.conversations[id];
    if (!conv || !window.confirm("Delete \"" + conv.title + "\"?")) return;
    delete data.conversations[id];
    delete errors[id];
    if (data.current === id) data.current = null;
    save();
    render();
  }

  /* Talking to the agent */

  function send(text) {
    text = text.trim();
    var conv = current();
    if (!text || (conv && pending[conv.id])) return;
    if (!conv) conv = createConversation(text);
    conv.messages.push({ role: "user", content: text, time: Date.now() });
    conv.updated = Date.now();
    save();
    el.input.value = "";
    autosize();
    requestReply(conv);
  }

  function requestReply(conv) {
    pending[conv.id] = true;
    delete errors[conv.id];
    render();

    var body = {
      model: conv.model,
      messages: conv.messages.map(function (m) { return { role: m.role, content: m.content }; })
    };
    fetch(config.apiUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    }).then(function (response) {
      return response.json().catch(function () { return {}; }).then(function (json) {
        if (!response.ok) throw new Error(json.error || errorMessage(response.status));
        return json;
      });
    }).then(function (json) {
      var message = { role: "assistant", content: json.reply || "", time: Date.now() };
      if (json.events && json.events.length) message.events = json.events;
      // The conversation may have been reloaded from storage (another tab) or deleted meanwhile
      var target = data.conversations[conv.id];
      if (target) {
        target.messages.push(message);
        target.updated = Date.now();
        save();
      }
    }).catch(function (error) {
      errors[conv.id] = error && error.message && error.message !== "Failed to fetch"
        ? error.message : "Could not reach the server. Check your connection and try again.";
    }).then(function () {
      delete pending[conv.id];
      render();
      if (!isTouch()) el.input.focus();
    });
  }

  function errorMessage(status) {
    if (status === 429) return "Too many messages. Please wait a moment and try again.";
    if (status === 404) return "This agent is no longer available. Start a new chat.";
    return "Something went wrong (error " + status + "). Please try again.";
  }

  function retry() {
    var conv = current();
    if (conv && !pending[conv.id]) requestReply(conv);
  }

  /* Rendering */

  function render() {
    renderHeader();
    renderSidebar();
    renderMessages();
    updateSendButton();
  }

  function renderHeader() {
    var conv = current();
    var model = conv ? conv.model : draftModel;
    var name = agentName(model);
    el.title.textContent = config.title;
    el["header-avatar"].textContent = initial(name);
    var subtitle = conv && pending[conv.id] ? name + " is typing…" : (config.agents.length > 1 ? name : "");
    el.subtitle.textContent = subtitle;
    el.subtitle.hidden = !subtitle;
  }

  function renderSidebar() {
    var list = el["conversation-list"];
    list.textContent = "";
    var groups = {};
    var order = [];
    sortedConversations().forEach(function (conv) {
      var group = dateGroup(conv.updated);
      if (!groups[group]) { groups[group] = []; order.push(group); }
      groups[group].push(conv);
    });
    order.forEach(function (group) {
      list.appendChild(make("div", "conversation-group", group));
      groups[group].forEach(function (conv) {
        var item = make("div", "conversation" + (conv.id === data.current ? " active" : ""));
        var open = make("button", "conversation-open", conv.title);
        open.type = "button";
        open.title = conv.title;
        open.addEventListener("click", function () { openConversation(conv.id); });
        var del = make("button", "conversation-delete");
        del.type = "button";
        del.setAttribute("aria-label", "Delete conversation");
        del.title = "Delete";
        del.appendChild(icon("M4 7h16M10 11v6M14 11v6M5 7l1 12a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2l1-12M9 7V4h6v3"));
        del.addEventListener("click", function () { deleteConversation(conv.id); });
        item.appendChild(open);
        item.appendChild(del);
        list.appendChild(item);
      });
    });
  }

  function dateGroup(time) {
    var day = 86400000;
    var today = new Date();
    today.setHours(0, 0, 0, 0);
    if (time >= today.getTime()) return "Today";
    if (time >= today.getTime() - day) return "Yesterday";
    if (time >= today.getTime() - 7 * day) return "Previous 7 days";
    if (time >= today.getTime() - 30 * day) return "Previous 30 days";
    return "Older";
  }

  function renderMessages() {
    var box = el.messages;
    var conv = current();
    var stick = box.scrollHeight - box.scrollTop - box.clientHeight < 80;
    box.textContent = "";

    if (!conv || !conv.messages.length) {
      shown = { id: null, count: 0 };
      box.appendChild(renderEmpty());
      return;
    }

    var name = agentName(conv.model);
    // Only messages added since the last render of this conversation fade in, not the whole history
    var firstNew = shown.id === conv.id ? shown.count : conv.messages.length;
    conv.messages.forEach(function (message, index) {
      if (message.role === "assistant" && config.showEvents && message.events) {
        box.appendChild(renderEvents(message.events));
      }
      var row = renderMessage(message, name);
      if (index >= firstNew) row.classList.add("new");
      box.appendChild(row);
    });
    shown = { id: conv.id, count: conv.messages.length };

    if (pending[conv.id]) {
      var typing = make("span", "typing");
      typing.setAttribute("aria-label", name + " is typing");
      typing.appendChild(make("span"));
      typing.appendChild(make("span"));
      typing.appendChild(make("span"));
      box.appendChild(wrapAssistant(typing, name, "new"));
    } else if (errors[conv.id]) {
      var bubble = make("div", "bubble");
      bubble.appendChild(make("span", null, errors[conv.id]));
      var button = make("button", "retry", "Retry");
      button.type = "button";
      button.addEventListener("click", retry);
      bubble.appendChild(button);
      var row = make("div", "message assistant error");
      row.setAttribute("role", "alert");
      row.appendChild(bubble);
      box.appendChild(row);
    }
    if (stick || pending[conv.id]) box.scrollTop = box.scrollHeight;
  }

  function renderEmpty() {
    var model = draftModel;
    var name = agentName(model);
    var empty = make("div", "empty");
    var avatar = make("span", "avatar", initial(name));
    avatar.setAttribute("aria-hidden", "true");
    empty.appendChild(avatar);
    empty.appendChild(make("h2", null, "Chat with " + name));
    empty.appendChild(make("p", null, "Send a message to start the conversation."));
    if (config.agents.length > 1) {
      var picker = make("div", "agent-picker");
      picker.setAttribute("role", "group");
      picker.setAttribute("aria-label", "Choose an agent");
      config.agents.forEach(function (agent) {
        var choice = make("button", "agent-choice", agent.name);
        choice.type = "button";
        choice.setAttribute("aria-pressed", String(agent.id === model));
        choice.addEventListener("click", function () {
          draftModel = agent.id;
          render();
          el.input.focus();
        });
        picker.appendChild(choice);
      });
      empty.appendChild(picker);
    }
    return empty;
  }

  function renderMessage(message, name) {
    if (message.role === "user") {
      var row = make("div", "message user");
      row.appendChild(make("div", "bubble", message.content));
      return row;
    }
    var bubble;
    if (config.markdown) {
      bubble = make("div", "bubble md");
      bubble.innerHTML = renderMarkdown(message.content);  // escaped by renderMarkdown
    } else {
      bubble = make("div", "bubble plain", message.content);
    }
    return wrapAssistant(bubble, name, null, true);
  }

  function wrapAssistant(content, name, extraClass, isBubble) {
    var row = make("div", "message assistant" + (extraClass ? " " + extraClass : ""));
    var avatar = make("span", "avatar small", initial(name));
    avatar.setAttribute("aria-hidden", "true");
    row.appendChild(avatar);
    if (isBubble) {
      row.appendChild(content);
    } else {
      var bubble = make("div", "bubble");
      bubble.appendChild(content);
      row.appendChild(bubble);
    }
    return row;
  }

  function renderEvents(events) {
    var box = make("div", "events");
    events.forEach(function (event) {
      var details = make("details", "event " + event.type);
      details.appendChild(make("summary", null, event.label));
      details.appendChild(make("pre", null, event.content));
      box.appendChild(details);
    });
    return box;
  }

  function updateSendButton() {
    var conv = current();
    el.send.disabled = !el.input.value.trim() || !!(conv && pending[conv.id]);
  }

  /* Markdown (a small, safe subset: everything is HTML-escaped before formatting) */

  function escapeHtml(text) {
    return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  function renderInline(text) {
    var tokens = [];  // inline code and links, kept aside so that no other formatting applies inside them
    function keep(html) {
      tokens.push(html);
      return "\u0000" + (tokens.length - 1) + "\u0000";
    }
    text = text.replace(/\u0000/g, "")
      .replace(/`([^`\n]+)`/g, function (_, code) { return keep("<code>" + escapeHtml(code) + "</code>"); })
      .replace(/\[([^\]\n]+)\]\(((?:https?:\/\/|mailto:)[^\s)]+)\)/g, function (_, label, url) {
        return keep("<a href=\"" + escapeHtml(url) + "\" target=\"_blank\" rel=\"noopener noreferrer\">" +
                    escapeHtml(label) + "</a>");
      });
    text = escapeHtml(text);
    text = text.replace(/\*\*(\S(?:[^*]*?\S)?)\*\*/g, "<strong>$1</strong>")
      .replace(/__(\S(?:[^_]*?\S)?)__/g, "<strong>$1</strong>")
      .replace(/(^|[^*\w])\*(\S(?:[^*\n]*?\S)?)\*(?!\w)/g, "$1<em>$2</em>")
      .replace(/(^|[^_\w])_(\S(?:[^_\n]*?\S)?)_(?!\w)/g, "$1<em>$2</em>")
      .replace(/~~(\S(?:[^~]*?\S)?)~~/g, "<del>$1</del>");
    function restore(html) {  // tokens may hold others (e.g. code in a link's text)
      return html.replace(/\u0000(\d+)\u0000/g, function (_, i) { return restore(tokens[Number(i)]); });
    }
    return restore(text);
  }

  function renderMarkdown(src) {
    var html = [];
    var lines = src.replace(/\r\n?/g, "\n").split("\n");
    var i = 0;
    var paragraph = [];

    function flushParagraph() {
      if (paragraph.length) html.push("<p>" + paragraph.map(renderInline).join("<br>") + "</p>");
      paragraph = [];
    }

    while (i < lines.length) {
      var line = lines[i];
      var match;
      if ((match = line.match(/^\s*(```|~~~)/))) {
        flushParagraph();
        var fence = match[1];
        var code = [];
        i++;
        while (i < lines.length && lines[i].trim().indexOf(fence) !== 0) code.push(lines[i++]);
        i++;
        html.push("<pre><code>" + escapeHtml(code.join("\n")) + "</code></pre>");
        continue;
      }
      if (!line.trim()) { flushParagraph(); i++; continue; }
      if ((match = line.match(/^(#{1,6})\s+(.*)$/))) {
        flushParagraph();
        var level = Math.min(match[1].length, 4);
        html.push("<h" + level + ">" + renderInline(match[2].replace(/\s+#+\s*$/, "")) + "</h" + level + ">");
        i++;
        continue;
      }
      if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) { flushParagraph(); html.push("<hr>"); i++; continue; }
      if (/^\s*>/.test(line)) {
        flushParagraph();
        var quote = [];
        while (i < lines.length && /^\s*>/.test(lines[i])) quote.push(lines[i++].replace(/^\s*>\s?/, ""));
        html.push("<blockquote>" + renderMarkdown(quote.join("\n")) + "</blockquote>");
        continue;
      }
      var listMatch = line.match(/^\s*([-*+]|\d+[.)])\s+/);
      if (listMatch) {
        flushParagraph();
        var ordered = /\d/.test(listMatch[1]);
        var itemPattern = ordered ? /^\s*\d+[.)]\s+/ : /^\s*[-*+]\s+/;
        var items = [];
        while (i < lines.length && lines[i].trim()) {
          if (itemPattern.test(lines[i])) {
            items.push(lines[i].replace(itemPattern, ""));
          } else if (items.length && /^\s+/.test(lines[i])) {
            items[items.length - 1] += "\n" + lines[i].trim();  // continuation of the previous item
          } else {
            break;
          }
          i++;
        }
        var start = ordered ? parseInt(line.trim(), 10) : 1;
        html.push((ordered ? "<ol" + (start !== 1 ? " start=\"" + start + "\"" : "") + ">" : "<ul>") +
          items.map(function (item) {
            return "<li>" + item.split("\n").map(renderInline).join("<br>") + "</li>";
          }).join("") + (ordered ? "</ol>" : "</ul>"));
        continue;
      }
      paragraph.push(line);
      i++;
    }
    flushParagraph();
    return html.join("");
  }

  /* Helpers */

  function make(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }

  function icon(path) {
    var svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("aria-hidden", "true");
    var p = document.createElementNS("http://www.w3.org/2000/svg", "path");
    p.setAttribute("d", path);
    svg.appendChild(p);
    return svg;
  }

  function initial(name) {
    var letter = (name || "?").trim().charAt(0);
    return letter ? letter.toUpperCase() : "?";
  }

  function isTouch() {
    return window.matchMedia && window.matchMedia("(pointer: coarse)").matches;
  }

  function autosize() {
    el.input.style.height = "auto";
    el.input.style.height = Math.min(el.input.scrollHeight, 200) + "px";
    updateSendButton();
  }

  function openDrawer() {
    el.app.classList.add("drawer-open");
    el.scrim.hidden = false;
  }

  function closeDrawer() {
    el.app.classList.remove("drawer-open");
    el.scrim.hidden = true;
  }

  /* Setup */

  document.title = config.title;
  if (embedded) el.app.classList.add("embedded");
  if (!config.multipleConversations) el.app.classList.add("single");
  if (inWidget && window.parent !== window) {
    el["close-widget"].hidden = false;
    el["close-widget"].addEventListener("click", function () {
      window.parent.postMessage({ type: "sdialog-chat:close" }, "*");
    });
  }

  el.composer.addEventListener("submit", function (event) {
    event.preventDefault();
    send(el.input.value);
  });
  el.input.addEventListener("keydown", function (event) {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      send(el.input.value);
    }
  });
  el.input.addEventListener("input", autosize);
  el["new-chat"].addEventListener("click", startNewChat);
  el["new-chat-side"].addEventListener("click", startNewChat);
  el["open-sidebar"].addEventListener("click", openDrawer);
  el["close-sidebar"].addEventListener("click", closeDrawer);
  el.scrim.addEventListener("click", closeDrawer);
  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") closeDrawer();
  });

  // Keep several tabs of the chat in sync
  window.addEventListener("storage", function (event) {
    if (event.key !== config.storageKey) return;
    data = load();
    render();
  });

  // Conversations with agents that are no longer served keep their history, new ones use a served agent
  var served = config.agents.map(function (agent) { return agent.id; });
  if (draftModel && current() && served.indexOf(current().model) !== -1) draftModel = current().model;

  render();
  if (!isTouch()) el.input.focus();
})();

/*
 * SDialog chat button: adds a floating button to any page that opens the agent's chat in a panel.
 *
 *   <script src="https://your-server:1333/chat/widget.js" defer></script>
 *
 * Optional attributes on the <script> element:
 *   data-position="left"         button on the bottom left corner (default: right)
 *   data-color="#0f766e"         button color (default: indigo)
 *   data-label="Chat with us"    accessible name of the button (default: "Open chat")
 *   data-open="true"             open the chat when the page loads
 *
 * SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
 * SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
 * SPDX-License-Identifier: MIT
 */
(function () {
  "use strict";

  var script = document.currentScript;
  if (!script || window.__sdialogChatWidget) return;
  window.__sdialogChatWidget = true;

  var chatUrl = script.src.replace(/\/widget\.js(\?.*)?$/, "") + "?embed=1&widget=1";
  var chatOrigin = new URL(chatUrl).origin;
  var side = script.getAttribute("data-position") === "left" ? "left" : "right";
  var color = script.getAttribute("data-color") || "#4f46e5";
  var label = script.getAttribute("data-label") || "Open chat";
  var CHAT_ICON = '<svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" ' +
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
    '<path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12z"/></svg>';
  var CLOSE_ICON = '<svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" ' +
    'stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"/></svg>';

  var style = document.createElement("style");
  style.textContent = [
    ".sdialog-chat-button{position:fixed;bottom:20px;" + side + ":20px;z-index:2147483646;width:58px;height:58px;",
    "border:0;border-radius:50%;background:" + color + ";color:#fff;cursor:pointer;display:flex;",
    "align-items:center;justify-content:center;box-shadow:0 6px 20px rgba(0,0,0,.25);",
    "transition:transform .15s ease-out,box-shadow .15s}",
    ".sdialog-chat-button:hover{transform:scale(1.06);box-shadow:0 8px 26px rgba(0,0,0,.3)}",
    ".sdialog-chat-button:focus-visible{outline:3px solid " + color + ";outline-offset:3px}",
    ".sdialog-chat-panel{position:fixed;bottom:90px;" + side + ":20px;z-index:2147483647;width:390px;",
    "height:min(640px,calc(100vh - 110px));border:0;border-radius:18px;overflow:hidden;background:#fff;",
    "box-shadow:0 16px 48px rgba(0,0,0,.28);opacity:0;transform:translateY(12px) scale(.98);",
    "transform-origin:bottom " + side + ";pointer-events:none;transition:opacity .18s,transform .18s}",
    ".sdialog-chat-panel.open{opacity:1;transform:none;pointer-events:auto}",
    ".sdialog-chat-panel iframe{width:100%;height:100%;border:0;display:block}",
    "@media (prefers-color-scheme:dark){.sdialog-chat-panel{background:#1a1c20}}",
    "@media (max-width:480px){.sdialog-chat-panel{inset:0;width:auto;height:auto;border-radius:0}",
    ".sdialog-chat-panel.open~.sdialog-chat-button{display:none}}",
    "@media (prefers-reduced-motion:reduce){.sdialog-chat-panel,.sdialog-chat-button{transition:none}}"
  ].join("");
  document.head.appendChild(style);

  var panel = document.createElement("div");
  panel.className = "sdialog-chat-panel";
  panel.setAttribute("role", "dialog");
  panel.setAttribute("aria-label", "Chat");
  panel.hidden = true;

  var button = document.createElement("button");
  button.type = "button";
  button.className = "sdialog-chat-button";
  button.setAttribute("aria-label", label);
  button.setAttribute("aria-expanded", "false");
  button.innerHTML = CHAT_ICON;

  var iframe = null;
  var isOpen = false;

  function setOpen(open) {
    isOpen = open;
    if (open && !iframe) {  // loaded on first open, so pages don't load the chat until it is used
      iframe = document.createElement("iframe");
      iframe.src = chatUrl;
      iframe.title = "Chat";
      iframe.allow = "clipboard-write";
      panel.appendChild(iframe);
    }
    if (open) panel.hidden = false;
    // Next frame, so that the opening transition runs after the panel is displayed
    requestAnimationFrame(function () { panel.classList.toggle("open", open); });
    if (!open) setTimeout(function () { if (!isOpen) panel.hidden = true; }, 200);
    button.setAttribute("aria-expanded", String(open));
    button.setAttribute("aria-label", open ? "Close chat" : label);
    button.innerHTML = open ? CLOSE_ICON : CHAT_ICON;
    if (open && iframe) iframe.focus();
  }

  button.addEventListener("click", function () { setOpen(!isOpen); });
  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape" && isOpen) setOpen(false);
  });
  window.addEventListener("message", function (event) {
    if (event.origin === chatOrigin && event.data && event.data.type === "sdialog-chat:close") setOpen(false);
  });

  function mount() {
    document.body.appendChild(panel);
    document.body.appendChild(button);
    if (script.getAttribute("data-open") === "true") setOpen(true);
  }
  if (document.body) mount(); else document.addEventListener("DOMContentLoaded", mount);
})();
